#!/bin/bash

set -euo pipefail

cd "${HOME}/fmri-fm/experiments/eval_v1"

taskid=$1

for modelid in {0..6}; do
    bash launch_eval.sh $modelid $taskid
done
