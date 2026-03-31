#!/usr/bin/env bash
#SBATCH --job-name=long_seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=sophont
#SBATCH --array=0-35

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="long_seq"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    T64_pt16_1/patch
    T64_pt16_2/patch
    T64_pt16_3/patch
    T64_pt16_4/patch
    T16_pt16_1/patch
    T16_pt16_2/patch
)

datasets=(
    abide_dx
    adhd200_dx
    adni_ad_vs_cn
    ppmi_dx
    aabc_age
    aabc_sex
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf="logistic"

model="schaefer400_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=2"

notes="long sequence ablation $key; eval v2 (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval_v2/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

uv run --no-sync python -W ignore -m fmri_fm_eval.main_logistic \
    $model \
    $repr \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
