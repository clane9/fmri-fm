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
    attn_pep-4
    cross_pep-4
    crossreg_reg-4_pep-4
)
decodings=(
    attn
    cross
    crossreg
)
regs=(
    1
    1
    4
)
key=${keys[JOBID]}
decoding=${decodings[JOBID]}
reg=${regs[JOBID]}
# prediction edge padding to prevent interpolation across patch edge
pep=4

name="${EXP_NAME}/${key}/pretrain"
config="${EXP_DIR}/config/pretrain.yaml"

notes="decoding ablation test (decoding=${decoding}, reg_tokens=${reg}, pred_edge_pad=${pep})"
overrides="model_kwargs.decoding=${decoding} model_kwargs.reg_tokens=${reg} model_kwargs.pred_edge_pad=${pep}"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${config}" \
    --overrides \
    $overrides \
    name="${name}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
