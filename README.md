# Introduction

This is the source code for HKU Capstone. The goal of this project is to implement a grammatical error correction model using [PyTorch](https://github.com/pytorch/pytorch) and [fairseq](https://github.com/pytorch/fairseq).

The initial version of our code is based on source code of [Copy-Augmented Architecture](https://github.com/zhawe01/fairseq-gec). Then we add more data and try some tricks for the baseline model. Some of them work and we make some progress in F0.5 and GLEU. Our code is more about an empirical study on applying NLP techniques to a university team capstone.

## Downloads
- Download necessary pre-processed data and pre-trained models, just like original code's [introduction](https://github.com/zhawe01/fairseq-gec/blob/master/README.md)

- Download BEA train\valid\test [dataset](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz)

## Completed and works
- Baseline model
  Fork original [code](https://github.com/zhawe01/fairseq-gec) and train the model for 9 epochs, starting with the parameters of pre-trained model. F0.5 score achieves 58.86 and GLEU only gets 52.6.
```
sh generate.sh \${device_id} \${experiment_name}
sh generate_jfleg_gleu.sh \${device_id} \${experiment_name}
```

- Add BEA training set
  F0.5 score improves from 58.86 to 59.76.
```
sh generate.sh \${device_id} \${experiment_name}
```

- Rerank with [Language Model](https://github.com/pytorch/fairseq/tree/master/examples/language_model)
  F0.5 score reaches 60.64 and GLEU reaches 55.50.
```
sh interactive.sh \${device_id} \${experiment_name}  #change interactive.py into interactive_rerank.py
```

- Use context-aware neural spellchecker
  F0.5 score slightly drops and GLEU rises to 60.18. The main reason is that JFLEG test set has lots of spell mistakes and spellchecker can help a lot.

- Add rule-based [system](https://github.com/Findus23/pyLanguagetool)
  We pick up some basic rules in LanguageTool and add them into our grammar checker pipeline. F0.5 score rises to 61.13 and GLEU reaches 61.13.
```
sh interactive_spellchecked_rule.sh \${device_id} \${experiment_name}
```

## Completed and not works
- Error Generation Experiment
  Basically, we got a synthetic dataset accompany with the natural corpus for training our grammar correction model and the error generation model is trained with Lang-8 corpus. The number of new sentences is about 500,000. The model did not perform very well probability due to the choice of seed corpus for error generation.

- Boost inference(or iterative decoding)
  The fluency score is computed by a combination of model's output score and language model's score. The sentence with the highest fluency score is selected for the next iteration. When the highest score of this iteration is no higher than that of the previous iteration, the iteration ends and the final result is output. Since boost results in smoother sentences sometimes changes the order and structure of sentences, F0.5 score and GLEU did not improve.

## Not completed
- Try different training data for error generation model, different seed corpus and number of new sentences.
- Optimize boost inference progress, like set different parameters for iterative number.
- Add multi-task learning, like sentence-level task and token-level task.
- Evaluation of BEA test set using F0.5 with new tricks.
- Prettify interactive mode webpage and add more function for Web API, like highlight the corrected error.

## Dependecies
The basic environment used for the project is Windows 10 64bit + Python 3.7.1 + CUDA 10.1 + pytorch 1.3.0.

Some preprocessing scripts and paragraph tokenize process needs the NLTK framework, which can be installed with this command:

```
pip install nltk
```

Once the NLTK framework is installed, the `punkt` dataset must also be downloaded:

```
python
>>> import nltk
>>> nltk.download('punkt')
```

Besides, we need add rule-based framework into our grammar error correction system. The framework is the python version of [LanguageTool](https://github.com/languagetool-org/languagetool):

```
pip install pylanguagetool
# or via pipenv
pipenv install pylanguagetool
```

## Capstone Team

This Capstone was completed as the final project for DASC 7600: Data Science Project. Our team's work distribution lists below:

* Diangu Xie: jobs distribution, baseline model training, add BEA datasets, language model research and test, build and maintain cloud server, web GUI, documentation.
* Binfan Zheng: create and maintain HKU's GPU server, build neural spellchecker, multi-task model training.
* Haohan Huang: error generation model construction and training, GLEU scoring function, research and build rule-based system.
* Peixin Gao: boost inference(iterative decoding), fluency scoring, re-ranking.

Some of the work described above is completed by more than one person. Therefore, the list is only a basic summary. Besides, a report is included in this repository and our Web UI is also available in another repository.

## Acknowledgments
Our code was modified from [fairseq](https://github.com/pytorch/fairseq) codebase. We use the same license as fairseq(-py).

