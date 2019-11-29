#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import sys
import torch


from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module
import os
import numpy as np
from collections import defaultdict
from fairseq.lm_scorer import LMScorer
import argparse
from spellchecker import SpellChecker
import string

from pylanguagetool import api


Batch = namedtuple('Batch', 'ids src_tokens src_lengths, src_strs')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

lm_path = r'./language_model/wiki103_fconv/wiki103.pt'
lm_databin = r'./language_model/data-bin'

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

class SpellChecking():
    def __init__(self, stc, lm_scorer):#st means sentence which need be checked by spellcheker
        self.stc = stc
        self.lm_scorer = lm_scorer
        self.checker = SpellChecker()
    def puncRemove(self, sentence):
        tranTemp = str.maketrans({key: None for key in string.punctuation})
        tgtSentence = sentence.translate(tranTemp)
        return tgtSentence
    def errorFind(self, sentence):
        #tgtSentence = self.puncRemove(sentence)
        tokenList = sentence.split(' ')
        posList = [] #Store the position of wrong words' positions
        numList = [] #Store the number of wrong words' candidates
        for tokenIndex in range(len(tokenList)):
            if self.checker.correction(tokenList[tokenIndex]) != tokenList[tokenIndex]: #Checking if this word is right
                posList.append(tokenIndex) #will be replaced by method searching for all candidates by language model 
                numList.append(len(self.checker.candidates(tokenList[tokenIndex])))
        return (posList, numList)
    def suggest(self, sentence):
        (posList, numList) = self.errorFind(sentence)
        lenInt = len(sentence.split(' '))
        for index in range(len(numList)):
            candiList = []
            lenNewInt = len(sentence.split(' '))
            lenGapInt = lenNewInt - lenInt
            #print(posList, numList)
            if lenGapInt != 0:
                lenInt = lenNewInt
                for pos in range(len(posList)):
                    posList[pos] += lenGapInt
            for num in range(numList[index]):
                tokenList = sentence.split(' ')
                tokenList[posList[index]]=list(self.checker.candidates(tokenList[posList[index]]))[num]
                newSentence = ' '.join(tokenList)
                candiList.append(newSentence) # Store candidates
            scoreDict = self.lm_scorer.score(candiList) # Score sentences
            maxInt = max(scoreDict, key=scoreDict.get) #find the max
            sentence = candiList[maxInt]
        return sentence


def sentence_check(src):
    # input a source sentence and correct the first error

    enable_rule_list1 = ['COMMA_COMPOUND_SENTENCE', 'EN_QUOTES', 'SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA', 
                         'DOUBLE_PUNCTUATION', 'COMMA_PARENTHESIS_WHITESPACE', 
                         'DELETE_SPACE', 'SENTENCE_WHITESPACE', 'DASH_RULE']
    enable_rule_list2 = ['PLURAL_VERB_AFTER_THIS', 'DOES_YOU', 'FEWER_LESS', 'UPPERCASE_SENTENCE_START',
                        'EN_A_VS_AN', 'EVERYDAY_EVERY_DAY', 'CONFUSION_OF_THESES_THESE', 'DO_ARTS',
                        'WHO_WHOM', 'THIS_NNS', 'THE_SUPERLATIVE', 'MENTION_ABOUT',
                        'USE_TO_VERB', 'LOT_OF', 'MANY_NN', 'A_UNCOUNTABLE',
                        'DOWN_SIDE', 'HAVE_PART_AGREEMENT', 'NODT_DOZEN',
                        'PHRASE_REPETITION', 'ADVISE_VBG', 'COMPARISONS_AS_ADJECTIVE_AS'
                        ]
    
    #res_dict = api.check(src, api_url='https://languagetool.org/api/v2/', lang='en-US')
    res_dict = api.check(src, api_url='http://localhost:8081/v2/', lang='en-US')
    res_matches = res_dict['matches']
    res_matches = [m for m in res_matches if len(m['replacements'])>0]
    res_matches = [m for m in res_matches if (m['rule']['id'] in enable_rule_list1) or (m['rule']['id'] in enable_rule_list2)]
    #res_matches = [m for m in res_matches if m['rule']['id'] in enable_rule_list2] # only use list 2 for generate
    if len(res_matches)==0:
        return None # no mistake detected
    match = res_matches[0]
    tmp_from = match['offset']; tmp_to = tmp_from + match['length']
    tgt = src[:tmp_from] + match['replacements'][0]['value'] + src[tmp_to:]
    return tgt, match['message'], match['rule']['id']


def capitalize_proper_non(src, nameList):
    srcList = src.split(' ')
    for i in range(len(srcList)):
        wordStr = srcList[i]
        if wordStr.capitalize() in nameList:
            srcList[i] = wordStr.capitalize()
    return ' '.join(srcList)  

    
def iter_check(src):
    # input a source sentence and return the correct one.
    tgt = src    
    msgList = []; ruleList = []
    tmp_res = sentence_check(src)
    count = 0 # max number of modificaition by rule
    while tmp_res and count <= 20:
        count += 1
        next_tgt, msg, rule = tmp_res
        tgt = next_tgt
        msgList.append(msg); ruleList.append(rule)
        tmp_res = sentence_check(tgt)
    # self-defined nameList    
    tgt = capitalize_proper_non(tgt, nameList = ['Facebook', 'I'])
    return tgt


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False, copy_ext_dict=args.copy_ext_dict).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            src_strs=[lines[i] for i in batch['id']],
        )


def main(args):
    import_user_module(args)
    lm_scorer = load_lm()
    check = SpellChecking('----Let us transform----',lm_scorer)
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )
    args.copy_ext_dict = getattr(_model_args, "copy_attention", False)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)
    if align_dict is None and args.copy_ext_dict:
        align_dict = {}

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    src_strs = []
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        lineStrSrc = inputs[0][:-1]
        inputs  = [check.suggest(lineStrSrc)]
        for batch in make_batches(inputs, args, task, max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            src_strs.extend(batch.src_strs)
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_strs[id],
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                hypo_str_rule = iter_check(hypo_str)
                print('H-{}\t{}\t{}\t{}'.format(id, hypo['score'], hypo_str, hypo_str_rule))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        # update running id counter
        start_id += len(results)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
