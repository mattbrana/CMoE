# CMoE

Implementation for the paper [CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference](https://arxiv.org/abs/2502.04416). 

## Dependencies

```bash
conda create -n cmoe python=3.11
conda activate cmoe
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install datasets==2.21.0
pip install transformers==4.47.1
pip install accelerate==1.2.1
pip install sentencepiece==0.2.0
pip install protobuf==5.29.2
pip install matplotlib==3.10.0
pip install lap==0.5.12
pip install peft==0.14.0
```
Note: please modify the version of some packages for your own environment.

## Quick Start

The included pipeline measures **GPU energy** (via NVML) and **PPL** for the
**dense** model and the **carved MoE** model, so you can directly compare
"before vs after" in [Weights & Biases](https://wandb.ai/).

### 1. Install + log in to W&B

```bash
pip install -r requirements.txt
wandb login
```

### 2. Download the Hugging Face model (one-time)

```bash
# Linux / macOS / Git Bash
bash run.sh download

# Windows PowerShell
.\run.ps1 download
```

This calls `download_model.py`, which fetches a snapshot via `huggingface_hub`.
Set `MODEL=...` to choose a different repo, and pass `--token hf_xxx` for
gated repos (or set `HF_TOKEN` in your env). The printed local path can be
re-used as `MODEL_PATH` if you prefer to keep the weights in a fixed folder.

### 3. Run the full energy comparison pipeline

```bash
# Linux / macOS / Git Bash
bash run.sh

# Windows PowerShell
.\run.ps1
```

The pipeline runs five W&B-logged phases in a single process:

| Phase | W&B `phase_name`         | What it measures                                     |
|-------|--------------------------|------------------------------------------------------|
| 0     | `dense_baseline_eval`    | PPL + GPU Joules of the **original dense** model     |
| 1     | `conversion`             | GPU Joules to **carve** the MoE                      |
| 2     | `post_conversion_eval`   | PPL + Joules of the **MoE before fine-tune**         |
| 3     | `finetune`               | GPU Joules during LoRA fine-tune                     |
| 4     | `post_finetune_eval`     | PPL + Joules of the **MoE after fine-tune**          |

In W&B look for:

- `phase/<name>_joules` summaries (one per phase)
- `ppl/dense_baseline_*`, `ppl/post_conversion_*`, `ppl/finetuned_*`
- `energy/gpu_power_w` and `energy/total_energy_j` time-series

PPL numbers are also written to `result_logs/<model>_<dataset>_*.txt`
(`dense_ppl`, `pre_ppl`, `ft_ppl`).

### 4. Customize

All knobs are environment variables in the launcher scripts:

```bash
MODEL=meta-llama/Meta-Llama-3-8B \
DATASET=wikitext2 \
NSHARED=2 NACTIVATED=2 NEXPERTS=16 \
NSAMPLES=2048 EPOCH=1 \
bash run.sh
```

To skip the dense baseline (e.g. if you've already measured it):

```bash
bash run.sh -- --skip-dense-baseline
```

You can still call `run_cmoe.py` directly:

```bash
python run_cmoe.py $MODEL_PATH wikitext2 \
    --nshared 2 --nactivated 2 --nexperts 16 \
    --nsamples 2048 --extra-lr 0.001 --bias-speed 0.001 --new-eval
```

## Evaluation

PPL eval runs automatically (in phases 0, 2, and 4 above).
For downstream tasks, add `--eval-zero` (implemented via
[Wanda](https://github.com/locuslab/wanda)).

## Cite

If you found this work useful, please consider citing:

```
@article{pei2025cmoe,
  title={CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference},
  author={Pei, Zehua and Zou, Lancheng and Zhen, Hui-Ling and Yu, Xianzhi and Liu, Wulong and Pan, Sinno Jialin and Yuan, Mingxuan and Yu, Bei},
  journal={arXiv preprint arXiv:2502.04416},
  year={2025}
}
```
