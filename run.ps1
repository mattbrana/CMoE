# CMoE energy comparison pipeline (Windows / PowerShell).
#
# Stages (each logged to W&B):
#   PHASE 0  dense_baseline_eval     dense model PPL + GPU energy
#   PHASE 1  conversion              MoE carving energy
#   PHASE 2  post_conversion_eval    MoE PPL + energy (training-free)
#   PHASE 3  finetune                LoRA fine-tuning energy
#   PHASE 4  post_finetune_eval      MoE PPL + energy (after fine-tune)
#
# Usage:
#   .\run.ps1 download              # one-time: pre-download the HF model
#   .\run.ps1                       # full pipeline
#   .\run.ps1 -- --skip-dense-baseline   # forward extra args to run_cmoe.py
#
# Override any variable from the environment, e.g.:
#   $env:MODEL = "meta-llama/Meta-Llama-3-8B"; .\run.ps1

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = 'Stop'

if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = '0' }

$Model         = if ($env:MODEL)         { $env:MODEL }         else { 'meta-llama/Llama-2-7b-hf' }
$Dataset       = if ($env:DATASET)       { $env:DATASET }       else { 'wikitext2' }
$NShared       = if ($env:NSHARED)       { $env:NSHARED }       else { '2' }
$NActivated    = if ($env:NACTIVATED)    { $env:NACTIVATED }    else { '2' }
$NExperts      = if ($env:NEXPERTS)      { $env:NEXPERTS }      else { '16' }
$NSamples      = if ($env:NSAMPLES)      { $env:NSAMPLES }      else { '64' }
$Epoch         = if ($env:EPOCH)         { $env:EPOCH }         else { '1' }
$ExtraLR       = if ($env:EXTRA_LR)      { $env:EXTRA_LR }      else { '0.001' }
$BiasSpeed     = if ($env:BIAS_SPEED)    { $env:BIAS_SPEED }    else { '0.001' }
$WandbProject  = if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { 'cmoe-energy' }

if ($ExtraArgs.Count -gt 0 -and $ExtraArgs[0] -eq 'download') {
    $rest = if ($ExtraArgs.Count -gt 1) { $ExtraArgs[1..($ExtraArgs.Count - 1)] } else { @() }
    python download_model.py $Model @rest
    exit $LASTEXITCODE
}

if ($ExtraArgs.Count -gt 0 -and $ExtraArgs[0] -eq '--') {
    $ExtraArgs = if ($ExtraArgs.Count -gt 1) { $ExtraArgs[1..($ExtraArgs.Count - 1)] } else { @() }
}

python run_cmoe.py $Model $Dataset `
    --new-eval `
    --nshared $NShared `
    --nactivated $NActivated `
    --nexperts $NExperts `
    --nsamples $NSamples `
    --epoch $Epoch `
    --extra-lr $ExtraLR `
    --bias-speed $BiasSpeed `
    --wandb-project $WandbProject `
    @ExtraArgs

exit $LASTEXITCODE
