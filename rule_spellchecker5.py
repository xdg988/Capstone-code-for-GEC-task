#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:49:04 2019

"""


from pylanguagetool import api
from collections import Counter

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
    #res_matches = [m for m in res_matches if (m['rule']['id'] in enable_rule_list1) or (m['rule']['id'] in enable_rule_list2)]
    res_matches = [m for m in res_matches if m['rule']['id'] in enable_rule_list2] # only use list 2 for generate
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
     


# read into the sentence before rule check
path = './rule_based/reranked_baseline.txt'
with open(path, 'r') as f:
    src = f.readlines()


# checking
tgt = [] # target sentences
msgs = [] # all_output messages for each sentence
rules = [] # all output rules for each sentence
for s in src:
    print(s)
    tmp_tgt, tmp_msg, tmp_rule = iter_check(s)
    tgt.append(tmp_tgt)
    msgs.append(tmp_msg)
    rules.append(tmp_rule)

# recording
path = './rule_based/reranked_ruled.txt'
with open(path, 'w') as f:
    f.writelines(tgt)    