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
#SBATCH --array=4-5

set -euo pipefail

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="long_seq"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

# rerunning mni cortex after fixing lr flip on mask
configs=(
    "T64_pt16_1|num_frames=64 t_patch_size=16 seed=5401"
    "T64_pt16_2|num_frames=64 t_patch_size=16 seed=5402"
    "T64_pt16_3|num_frames=64 t_patch_size=16 seed=5403"
    "T64_pt16_4|num_frames=64 t_patch_size=16 seed=5404"
    "T16_pt16_1|num_frames=16 t_patch_size=16 seed=5401"
    "T16_pt16_2|num_frames=16 t_patch_size=16 seed=5402"
    "T16_pt16_3|num_frames=16 t_patch_size=16 seed=5403"
    "T16_pt16_4|num_frames=16 t_patch_size=16 seed=5404"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="long sequence ablation $name (${overrides})"

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
