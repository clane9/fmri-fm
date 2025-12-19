#!/bin/bash

if [[ -z $1 || $1 == "-h" || $1 == "--help" ]]; then
    echo "launch_pretrain.sh JOBID"
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

EXP_NAME="decoders"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/checkpoints"

# fill with the name of your home folder on lightning
SHARE_USER=${SHARE_USER:-volunteer}
SHARE_DIR="/teamspace/gcs_folders/share/fmri-fm/${SHARE_USER}"

# save output to persistent shared storage
SHARE_OUT_DIR="${SHARE_DIR}/${OUT_DIR}"
mkdir -p ${SHARE_OUT_DIR} 2>/dev/null
ln -sn ${SHARE_OUT_DIR} ${OUT_DIR} 2>/dev/null

keys=(
    attn_reg-1
    attn_reg-4
    cross_reg-1
    cross_reg-4
    crossreg_reg-1
    crossreg_reg-4
    crossreg_reg-16
)
decodings=(
    attn
    attn
    cross
    cross
    crossreg
    crossreg
    crossreg
)
regs=(
    1
    4
    1
    4
    1
    4
    16
)
key=${keys[JOBID]}
decoding=${decodings[JOBID]}
reg=${regs[JOBID]}

name="${EXP_NAME}/${key}/pretrain"
config="${EXP_DIR}/config/pretrain.yaml"

notes="decoding ablation test (decoding=${decoding}, reg_tokens=${reg})"
overrides="model_kwargs.decoding=${decoding} model_kwargs.reg_tokens=${reg}"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${config}" \
    --overrides \
    $overrides \
    name="${name}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
