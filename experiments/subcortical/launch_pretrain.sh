#!/usr/bin/env bash
#SBATCH --job-name=subcortical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=sophont
#SBATCH --array=0-11

set -euo pipefail

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="subcortical"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

# rerunning mni cortex after fixing lr flip on mask
configs=(
    "schaefer400_lr3e-4_1|input_space=schaefer400 base_lr=3e-4 seed=5401"
    "schaefer400_lr3e-4_2|input_space=schaefer400 base_lr=3e-4 seed=5402"
    "schaefer400_lr3e-4_3|input_space=schaefer400 base_lr=3e-4 seed=5403"
    "schaefer400_lr3e-4_4|input_space=schaefer400 base_lr=3e-4 seed=5404"
    "schaefer400_tians3_lr3e-4_1|input_space=schaefer400_tians3 base_lr=3e-4 seed=5401"
    "schaefer400_tians3_lr3e-4_2|input_space=schaefer400_tians3 base_lr=3e-4 seed=5402"
    "schaefer400_tians3_lr3e-4_3|input_space=schaefer400_tians3 base_lr=3e-4 seed=5403"
    "schaefer400_tians3_lr3e-4_4|input_space=schaefer400_tians3 base_lr=3e-4 seed=5404"
    "a424_lr3e-4_1|input_space=a424 base_lr=3e-4 seed=5401"
    "a424_lr3e-4_2|input_space=a424 base_lr=3e-4 seed=5402"
    "a424_lr3e-4_3|input_space=a424 base_lr=3e-4 seed=5403"
    "a424_lr3e-4_4|input_space=a424 base_lr=3e-4 seed=5404"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="subcortical ablation $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 5 ))

# for debugging
# overrides="${overrides} debug=true wandb=false"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
