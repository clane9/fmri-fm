#!/bin/bash

if [[ -z $1 || $1 == "-h" || $1 == "--help" ]]; then
    echo "launch_pretrain_1.sh JOBID"
    exit
fi

JOBID=$1

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="pretrain_mix"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/checkpoints"

# fill with the name of your home folder on lightning
SHARE_USER=${SHARE_USER:-volunteer}
SHARE_DIR="/teamspace/gcs_folders/share/fmri-fm/${SHARE_USER}"

# save output to persistent shared storage
SHARE_OUT_DIR="${SHARE_DIR}/${OUT_DIR}"
mkdir -p ${SHARE_OUT_DIR} 2>/dev/null
ln -sn ${SHARE_OUT_DIR} ${OUT_DIR} 2>/dev/null

names=(
    hcp_n1800_ep200
    ukbb_n1800_ep200
    hcp_ukbb_n3600_ep200
)
configs=(
    hcp_n1800
    ukbb_n1800
    hcp_ukbb_n3600
)

name=${names[JOBID]}
config=${configs[JOBID]}

name="${EXP_NAME}/${name}/pretrain"
config="${EXP_DIR}/config/pretrain_${config}.yaml"

notes="hcp ukbb mix pretraining run; 200 epoch schedule."

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${config}" \
    --overrides \
    name="${name}" \
    epochs=200 \
    output_dir="${OUT_DIR}"
