import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from CMoE_utils import *
from CMoE_model import *
from zero_eval import *
from sft_utils import simple_sft

# ── WANDB ENERGY TRACKING ──────────────────────────────────────────
import wandb
from energy_tracker import EnergyTracker
# ───────────────────────────────────────────────────────────────────

DEV = torch.device('cuda:0')
def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model


def _restore_full_gpu_placement(model):
    """`cmoe_ppl_eval` offloads layers to CPU. Put everything back on the GPU
    so the carving / fine-tuning stages can run."""
    model.cuda()
    model.model.layers.cuda()
    torch.cuda.empty_cache()


def cmoe_sequential(model, dataloader, dev, args, tracker=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 8
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    moe_outs = torch.zeros_like(inps)
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    model.cuda()
    layers.cuda()

    inp = copy.deepcopy(inps[0])

    if tracker is None:
        tracker = EnergyTracker(gpu_index=0)
        owns_tracker = True
    else:
        owns_tracker = False

    # ── PHASE 0: Dense baseline PPL + energy ──────────────────────
    # Runs the SAME eval code path that we use after carving, so the
    # before/after numbers are directly comparable.
    dense_ppl = []
    if not getattr(args, 'skip_dense_baseline', False):
        print('Dense baseline PPL (pre-conversion):')
        tracker.start()
        eval_datasets = ['wikitext2', 'c4-new']
        for dataset in eval_datasets:
            _, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            ppl_i = cmoe_ppl_eval(model, testloader, DEV, dataset, args)
            dense_ppl.append(f"{dataset}: {ppl_i}")
            wandb.log({f"ppl/dense_baseline_{dataset}": ppl_i})
        tracker.stop(phase_name="dense_baseline_eval")
        # cmoe_ppl_eval offloaded layers to CPU; put them back for carving.
        _restore_full_gpu_placement(model)
        layers = model.model.layers
    # ──────────────────────────────────────────────────────────────

    # ── PHASE 1: MoE Carving ───────────────────────────────────────
    tracker.start()

    carve_inp = copy.deepcopy(inp)
    for layer in tqdm(layers, desc = 'Carving MoE layers...'):
        moe_out = construct_moe(layer, 
            carve_inp, 
            attention_mask, 
            position_ids,
            n_experts = args.nexperts,
            n_activated = args.nactivated,
            n_shared = args.nshared,
            args = args
        )
        carve_inp = moe_out

    tracker.stop(phase_name="conversion")
    # ──────────────────────────────────────────────────────────────

    tick_1 = time.time()

    # ── PHASE 2: Post-conversion (training-free) PPL eval ─────────
    tracker.start()

    print('Post-conversion (training-free) PPL:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, DEV, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
        wandb.log({
            f"ppl/post_conversion_{dataset}": ppl_i,
            f"ppl/training_free_{dataset}": ppl_i,
        })

    tracker.stop(phase_name="post_conversion_eval")
    # ──────────────────────────────────────────────────────────────
    
    tick_2 = time.time()

    # ── PHASE 3: LoRA Fine-tuning ──────────────────────────────────
    tracker.start()

    for layer in layers:
        layer.mlp.cus_training = True

    model.cuda()
    model = simple_sft(model, args, epoch = args.epoch)

    for layer in layers:
        layer.mlp.cus_training = False

    tracker.stop(phase_name="finetune")
    # ──────────────────────────────────────────────────────────────

    if owns_tracker:
        tracker.shutdown()

    model.eval()
    model.config.use_cache = use_cache

    return model, tick_1, tick_2, pre_ppl, dense_ppl, tracker

@torch.no_grad()
def cmoe_ppl_eval(model, testenc, dev, eval_set, args):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers)), desc= 'Processing...'):

        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()

def save_results(file_name, results):
    if results is not str:
        results = str(results)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--extra-lr',
        type=float, default=0.001, 
        help='Initial learning rate for extra scale for router.'
    )
    parser.add_argument(
        '--k-act', type=int, default=10,
        help='TopK number for the ATopK. K_a in paper.'
    )
    parser.add_argument(
        '--bias-speed',
        type=float, default=0.001, 
        help='Bias update speed for load balancing. Gamma in paper.'
    )
    parser.add_argument(
        '--nexperts', type=int, default=16,
        help='Total number of experts. N in paper.'
    )
    parser.add_argument(
        '--nactivated', type=int, default=2,
        help='Number of activated routed experts.'
    )
    parser.add_argument(
        '--nshared', type=int, default=2,
        help='Number of shared experts.'
    )
    parser.add_argument(
        '--epoch', type=int, default=1,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(
        '--sft-bsz', type=int, default=2,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(
        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )
    parser.add_argument(
        '--skip-dense-baseline', action='store_true',
        help='Skip the dense (pre-conversion) PPL + energy baseline.'
    )
    parser.add_argument(
        '--wandb-project', type=str, default='cmoe-energy',
        help='W&B project name for the energy comparison run.'
    )
    parser.add_argument(
        '--wandb-run-name', type=str, default=None,
        help='Optional W&B run name. Defaults to a descriptive auto name.'
    )

    args = parser.parse_args()

    model_short = args.model.split('/')[-1] if args.model else 'model'
    default_run_name = (
        f"{model_short}_{args.dataset}_S{args.nshared}A{args.nactivated}"
        f"E{args.nexperts}_n{args.nsamples}"
    )

    # ── INIT WANDB ────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or default_run_name,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "nshared": args.nshared,
            "nactivated": args.nactivated,
            "nexperts": args.nexperts,
            "nsamples": args.nsamples,
            "epoch": args.epoch,
            "extra_lr": args.extra_lr,
            "bias_speed": args.bias_speed,
            "skip_dense_baseline": args.skip_dense_baseline,
        }
    )
    # ─────────────────────────────────────────────────────────────

    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)
    model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tracker = EnergyTracker(gpu_index=0)

    tick = time.time()
    carved_model, tick_1, tick_2, pre_ppl, dense_ppl, tracker = cmoe_sequential(
        model, dataloader, DEV, args, tracker=tracker
    )
    rt_construct = tick_1 - tick
    extra_time = tick_2 - tick_1
    rt = time.time() - tick - extra_time
    print("Runtime of training-free construction: ", rt_construct)
    print("Runtime of fine-tuning construction: ", rt)

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if 'llama-3' in args.model.lower():
        name = "meta-llama/Meta-Llama-3-8B"
    else:
        name = "meta-llama/Llama-2-7b-hf"

    # ── PHASE 4: Post-finetune MoE baseline (energy + PPL) ─────────
    tracker.start()

    datasets = ['wikitext2', 'c4-new']
    ppl = []
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(carved_model, testloader, DEV, eval_set, args)
        ppl.append(f"{dataset}: {ppl_i}")

    tracker.stop(phase_name="post_finetune_eval")
    # ──────────────────────────────────────────────────────────────

    model_name = args.model.split("/")[-1]
    file_name = f"{model_name}_{args.dataset}_{args.nsamples}_epoch_{args.epoch}_S{args.nshared}_A{args.nactivated}_E{args.nexperts}.txt"
    dir_path = os.path.join('./result_logs', args.prefix) if args.prefix is not None else './result_logs'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.join(dir_path, file_name)

    save_results(file_name, f"dense_ppl: {str(dense_ppl)}")
    save_results(file_name, f"pre_ppl: {str(pre_ppl)}")
    save_results(file_name, f"ft_ppl: {str(ppl)}")
    save_results(file_name, f"runtime_construct: {rt_construct}")
    save_results(file_name, f"runtime_all: {rt}")

    # ── LOG FINAL RESULTS TO WANDB ─────────────────────────────────
    for entry in ppl:
        dataset_name, ppl_val = entry.split(": ")
        wandb.log({f"ppl/finetuned_{dataset_name.strip()}": float(ppl_val)})

    wandb.log({
        "runtime/construction_sec": rt_construct,
        "runtime/total_sec": rt,
    })

    tracker.shutdown()
    wandb.finish()
    # ──────────────────────────────────────────────────────────────

    if args.eval_zero:
        task_list = ["winogrande"]
        results_1 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=5)
        save_results(file_name, results_1)

        task_list = ["arc_challenge"]
        results_2 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=25)
        save_results(file_name, results_2)

        task_list = ["hellaswag"]
        results_3 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=10)
        save_results(file_name, results_3)

        task_list = ["sciq","piqa"]
        results_4 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=0)
        save_results(file_name, results_4)

        task_list = ["boolq"]
        results_5 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=32)
        save_results(file_name, results_5)


    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)


