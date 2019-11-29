import sys
import os
import numpy as np
from collections import defaultdict
from fairseq.lm_scorer import LMScorer
import argparse
import copy

lm_path = r'./language_model/adaptive_lm_gbw_huge/model.pt'
lm_databin = r'./language_model/adaptive_lm_gbw_huge/data-bin'

def load_lm(lm_path=lm_path, lm_databin=lm_databin):
    args = argparse.Namespace(
        path=lm_path, data=lm_databin,
        fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0,
        fp16_scale_window=None, fpath=None, future_target=False,
        gen_subset='test', lazy_load=False, log_format=None, log_interval=1000,
        max_sentences=None, max_tokens=None, memory_efficient_fp16=False,
        min_loss_scale=0.0001, model_overrides='{}', no_progress_bar=True,
        num_shards=1, num_workers=0, output_dictionary_size=-1,
        output_sent=False, past_target=False,
        quiet=True, raw_text=False, remove_bpe=None, sample_break_mode=None,
        seed=1, self_target=False, shard_id=0, skip_invalid_size_inputs_valid_test=False,
        task='language_modeling', tensorboard_logdir='', threshold_loss_scale=None,
        tokens_per_sample=1024, user_dir=None, cpu=False)
    return LMScorer(args)

def sort_list(line_list):
    tmp = [l[0] for l in line_list]
    score_dict = lm_scorer.score(tmp)
    for i in range(len(tmp)):
        line_list[i][1] = float(line_list[i][1]) + float(score_dict[i]) / 10
    line_list.sort(key=lambda k: k[1], reverse=True)
    tmp1 = [l[0] for l in line_list]
    return tmp1

if len(sys.argv) != 3:
    print('Usage: <beam> <filename>')
    exit(-1)
beam = int(sys.argv[1])
filename = sys.argv[2]
lm_scorer = load_lm()

x = sys.stdin.readlines()
hypo_dict = defaultdict(list)
for raw_line in x:
    raw_line_array = raw_line.strip().split('\t')
    hypo_dict[int(raw_line_array[0][2:])].append([raw_line_array[2], raw_line_array[1]])

ids = list(hypo_dict.keys())
line_lists = list(hypo_dict.values())

line_lists_new = []
for line_list in line_lists:
    line_lists_new.append(sort_list(line_list))

# sort line_lists by ids
idx = np.array(ids).argsort()

ofile = open(filename, 'w')
for line in np.array(line_lists_new)[idx]:
    ofile.write(line[0] + "\n")
ofile.close()
