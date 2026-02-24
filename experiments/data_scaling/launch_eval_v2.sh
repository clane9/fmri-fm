#!/usr/bin/env bash
#SBATCH --job-name=data_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2
#SBATCH --account=training
#SBATCH --array=0-19

set -euo pipefail

export OMP_NUM_THREADS=8

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="data_scaling"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    n100_1/patch/attn
    n200_1/patch/attn
    n400_1/patch/attn
    n800_1/patch/attn
    n1600_1/patch/attn
    n100_2/patch/attn
    n200_2/patch/attn
    n400_2/patch/attn
    n800_2/patch/attn
    n1600_2/patch/attn
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
# nb using best checkpoint not last, since the small data models are overfit
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-best.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=${bs} accum_iter=2"

notes="data scaling experiment $key; eval v2 (${dataset} ${repr} ${clf})"

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
