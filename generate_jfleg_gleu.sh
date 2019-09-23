#!/usr/bin/env bash
source ./config.sh

copy_params='--copy-ext-dict'
common_params='--source-lang src --target-lang tgt
--padding-factor 1
--srcdict ./dicts/dict.src.txt
--joined-dictionary
'

rm -rf ./out/jfleg_data_raw

python preprocess.py \
$common_params \
$copy_params \
--testpref data/jfleg_test \
--destdir ./out/jfleg_data_raw \
--output-format raw


set -e

ema='ema'

cp ./out/jfleg_data_raw/test.src-tgt.src ./out/jfleg_data_raw/test.src-tgt.src.old

rm -rf ./out/jfleg_data_raw/test.src-tgt.src ./out/jfleg_data_raw/test.src-tgt.tgt
python gec_scripts/split.py ./out/jfleg_data_raw/test.src-tgt.src.old ./out/jfleg_data_raw/test.src-tgt.src ./out/jfleg_data_raw/test.idx
cp ./out/jfleg_data_raw/test.src-tgt.src ./out/jfleg_data_raw/test.src-tgt.tgt

epochs='_last'
rm -rf $RESULT/jfleg_output/
mkdir $RESULT/jfleg_output/

for epoch in ${epochs[*]}; do
    if [ -f $RESULT/jfleg_output/jfleg$ema$exp_$epoch.log ]; then
        continue
    fi
    echo $epoch

    CUDA_VISIBLE_DEVICES=$device python generate.py ./out/jfleg_data_raw \
    --path $MODELS/checkpoint$ema$epoch.pt \
    --beam 12 \
    --nbest 12 \
    --gen-subset test \
    --max-tokens 6000 \
    --no-progress-bar \
    --raw-text \
    --batch-size 128 \
    --print-alignment \
    --max-len-a 0 \
    --no-early-stop \
    --copy-ext-dict --replace-unk \
    > $RESULT/jfleg_output/output$ema$epoch.nbest.txt 

    cat $RESULT/jfleg_output/output$ema$epoch.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 $RESULT/jfleg_output/output$ema$epoch.txt.split

    python ./gec_scripts/revert_split.py $RESULT/jfleg_output/output$ema$epoch.txt.split ./out/jfleg_data_raw/test.idx > $RESULT/jfleg_output/output$ema$epoch.txt

#    tail -n 1 $RESULT/jfleg_output/jfleg$ema$exp_$epoch.log
done

python ./out/jfleg-master/eval/gleu.py -r ./out/jfleg-master/test/test.ref[0-3] -s ./out/jfleg-master/test/test.src --hyp $RESULT/jfleg_output/outputema_last.txt

