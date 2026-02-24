#!/usr/bin/env bash
#SBATCH --job-name=decoders
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2
#SBATCH --account=training
#SBATCH --array=0-43

set -euo pipefail

export OMP_NUM_THREADS=8

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="decoders"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    attn_reg1/reg/linear
    cross_reg1/reg/linear
    crossreg_reg1/reg/linear
    crossreg_reg4/reg/linear
    crossreg_reg16/reg/linear
    attn_reg1_pep4/reg/linear
    cross_reg1_pep4/reg/linear
    crossreg_reg4_pep4/reg/linear
    attn_reg1/patch/attn
    cross_reg1/patch/attn
    crossreg_reg1/patch/attn
    crossreg_reg4/patch/attn
    crossreg_reg16/patch/attn
    attn_reg1_pep4/patch/attn
    cross_reg1_pep4/patch/attn
    crossreg_reg4_pep4/patch/attn
    crossreg_reg4/patch/attn
    crossreg_reg16/patch/attn
    crossreg_reg4/reg/attn
    crossreg_reg16/reg/attn
    crossreg_reg1_pep4/patch/attn
    crossreg_reg1_pep4/reg/linear
)

datasets=(
    hcpya_task21
    nsd_cococlip
)
batch_sizes=(
    64
    64
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

model="flat_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=${bs} accum_iter=2 classifier_kwargs.xavier_init=false classifier_kwargs.norm=false"

notes="decoder ablations $key; eval v2 (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval_v2/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

# add small delay between jobs
# sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

uv run --no-sync python -m fmri_fm_eval.main_probe \
    $model \
    $repr \
    $clf \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
