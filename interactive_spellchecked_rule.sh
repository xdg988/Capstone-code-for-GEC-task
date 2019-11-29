#!/usr/bin/env bash
source ./config.sh

copy_params='--copy-ext-dict --replace-unk'
if $WO_COPY; then
    copy_params='--replace-unk'
fi

beam=12

CUDA_VISIBLE_DEVICES=0 python interactive_checked_rule_hhh.py $DATA_RAW \
--path out/modelsV3/checkpointema_last.pt \
--beam $beam \
--nbest $beam \
--no-progress-bar \
--print-alignment \
$copy_params

#--replace-unk ./data/bin/alignment.src-tgt.txt \
#--path $MODELS/checkpointema1.pt \
