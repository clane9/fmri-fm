#!/usr/bin/env bash
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --array=0-8

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="finetune"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

export NSD_ROOT="/data/connor/fmri-fm-eval/datasets/NSD/data/arrow"

configs=(
    "lora|1e-5"
    "lora|3e-5"
    "lora|1e-4"
    "partial|1e-5"
    "partial|3e-5"
    "partial|1e-4"
    "full|1e-5"
    "full|3e-5"
    "full|1e-4"
    "scratch|1e-5"
    "scratch|3e-5"
    "scratch|1e-4"
)

datasets=(
    nsd_cococlip
)

num_configs=${#configs[@]}
datasetid=$(($SLURM_ARRAY_TASK_ID / $num_configs))
configid=$(($SLURM_ARRAY_TASK_ID % $num_configs))

dataset=${datasets[datasetid]}

model="flat_mae"
ckpt_path="experiments/input_space_v2/output/input_space_v2/flat_lr1e-3_1/pretrain/checkpoint-last.pth"
repr="patch"
clf="attn"

config=${configs[configid]}
key=$(echo $config | cut -d '|' -f 1)
lr=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/finetune_${key}.yaml"

name="${EXP_NAME}/${key}_lr${lr}/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

overrides="model_kwargs.ckpt_path=${ckpt_path} lr=${lr} wandb=false"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
# sleep $(( configid * 5 ))

uv run --no-sync python -m fmri_fm_eval.main_finetune \
    $model \
    $repr \
    $clf \
    $dataset \
    --config $base_config \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    $overrides
