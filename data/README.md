# Parsing Data Generation
We performed the same data generation method as in ([Constituency Parsing with a Self-Attentive Encoder](https://aclanthology.org/P18-1249/)). Detailed descriptions can be found in ([their github](https://github.com/nikitakit/self-attentive-parser/tree/master/data)).
## English WSJ

1. Place a copy of the Penn Treebank
([LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)) in `data/raw/treebank_3`.
After doing this, `data/raw/treebank_3/parsed/mrg/wsj` should have folders
named `00`-`24`.
2. Place a copy of the revised Penn Treebank
([LDC2015T13](https://catalog.ldc.upenn.edu/LDC2015T13)) in
`data/raw/eng_news_txt_tbnk-ptb_revised`.
3. Ensure that the active version of Python is Python 3 and has `nltk` and
`pytokenizations` installed.
4. `cd data/wsj && ./build_corpus.sh`


## Chinese Treebank (CTB 5.1)

This prepares the standard Chinese constituency parsing split, following recent papers such as [Liu and Zhang (2017)](https://www.aclweb.org/anthology/Q17-1004/).

### Instructions

1. Place a copy of the Chinese Treebank 5.1
([LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01)) in `data/raw/ctb5.1_507K`.
2. Ensure that the active version of Python is Python 3 and has `nltk` installed.
3. `cd data/ctb_5.1 && ./build_corpus.sh`
