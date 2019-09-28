#!/usr/bin/env bash
source ./config.sh

set -e

ema='ema'

rm -rf $DATA_RAW/test1.src-tgt.src $DATA_RAW/test1.src-tgt.tgt
python gec_scripts/split.py $DATA_RAW/test1.src-tgt.src.old $DATA_RAW/test1.src-tgt.src $DATA_RAW/test1.idx
cp $DATA_RAW/test1.src-tgt.src $DATA_RAW/test1.src-tgt.tgt

epochs='_last'
for epoch in ${epochs[*]}; do
    if [ -f $RESULT/m2score1$ema$exp_$epoch.log ]; then
        continue
    fi
    echo $epoch

    CUDA_VISIBLE_DEVICES=$device python generate.py $DATA_RAW \
    --path $MODELS/checkpoint$ema$epoch.pt \
    --beam 12 \
    --nbest 12 \
    --gen-subset test1 \
    --max-tokens 6000 \
    --no-progress-bar \
    --raw-text \
    --batch-size 128 \
    --print-alignment \
    --max-len-a 0 \
    --no-early-stop \
    --copy-ext-dict --replace-unk \
    > $RESULT/output1$ema$epoch.nbest.txt

    cat $RESULT/output1$ema$epoch.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 $RESULT/output1$ema$epoch.txt.split

    python ./gec_scripts/revert_split.py $RESULT/output1$ema$epoch.txt.split $DATA_RAW/test1.idx > $RESULT/output1$ema$epoch.txt
done
