# BNA Parser

## Contents
1. [Requirements](#Requirements)
2. [Pre-trained model](#Pre-trained model)
3. [Training](#training)
4. [Experiment results](#Experiment results)
5. [Citation](#citation)
6. [Credits](#credits)

## Requirements



## Training

Training requires cloning this repository from GitHub. While the model code in `src/benepar` is distributed in the `benepar` package on PyPI, the training and evaluation scripts directly under `src/` are not.

#### Software Requirements for Training
* Python 3.7 or higher.
* [PyTorch](http://pytorch.org/) 1.6.0, or any compatible version.
* All dependencies required by the `benepar` package, including: [NLTK](https://www.nltk.org/) 3.2, [torch-struct](https://github.com/harvardnlp/pytorch-struct) 0.4, [transformers](https://github.com/huggingface/transformers) 4.3.0, or compatible.
* [pytokenizations](https://github.com/tamuhey/tokenizations/) 0.7.2 or compatible.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.

### Training Instructions

A new model can be trained using the command `python src/main.py train ...`. Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/wsj/train_02-21.LDC99T42`
`--train-path-text` | Optional non-destructive tokenization of the training data | Guess raw text; see `--text-processing`
`--dev-path` | Path to development trees | `data/wsj/dev_22.LDC99T42`
`--dev-path-text` | Optional non-destructive tokenization of the development data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--batch-size` | Number of examples per training update | 32
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--numpy-seed` | NumPy random seed | Random
`--use-pretrained` | Use pre-trained encoder | Do not use pre-trained encoder
`--pretrained-model` | Model to use if `--use-pretrained` is passed. May be a path or a model id from the [HuggingFace Model Hub](https://huggingface.co/models)| `bert-base-uncased`
`--predict-tags` | Adds a part-of-speech tagging component and auxiliary loss to the parser | Do not predict tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-encoder` | Use learned transformer layers on top of pre-trained model or CharLSTM | Do not use extra transformer layers
`--num-layers` | Number of transformer layers to use if `--use-encoder` is passed | 8
`--encoder-max-len` | Maximum sentence length (in words) allowed for extra transformer layers | 512

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--predict-tags` (for boolean parameters that default to False), or `--no-XXX` (for boolean parameters that default to True).

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

Prior to training the parser, you will first need to obtain appropriate training data. We provide [instructions on how to process standard datasets like PTB, CTB, and the SMPRL 2013/2014 Shared Task data](data/README.md). After following the instructions for the English WSJ data, you can use the following command to train an English parser using the default hyperparameters:

```
python src/main.py train --use-pretrained --model-path-base models/en_bert_base
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for more examples of good hyperparameter choices.

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path` | Path of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-text` | Optional non-destructive tokenization of the test data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel (a GPU does not have enough memory to process the full dataset in one batch) | 500
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--output-path` | Path to write predicted trees to (use `"-"` for stdout). | Do not save predicted trees
`--no-predict-tags` | Use gold part-of-speech tags when running EVALB. This is the standard for publications, and omitting this flag may give erroneously high F1 scores. | Use predicted part-of-speech tags for EVALB, if available

As an example, you can evaluate a trained model using the following command:
```
python src/main.py test --model-path models/en_bert_base_dev=*.pt
```

### Exporting Models for Inference

The `benepar` package can directly use saved checkpoints by replacing a model name like `benepar_en3` with a path such as `models/en_bert_base_dev_dev=95.67.pt`. However, releasing the single-file checkpoints has a few shortcomings:
* Single-file checkpoints do not include the tokenizer or pre-trained model config. These can generally be downloaded automatically from the HuggingFace model hub, but this requires an Internet connection and may also (incidentally and unnecessarily) download pre-trained weights from the HuggingFace Model Hub
* Single-file checkpoints are 3x larger than necessary, because they save optimizer state

Use `src/export.py` to convert a checkpoint file into a directory that encapsulates everything about a trained model. For example,
```
python src/export.py export \
  --model-path models/en_bert_base_dev=*.pt \
  --output-dir=models/en_bert_base
```

When exporting, there is also a `--compress` option that slightly adjusts model weights, so that the output directory can be compressed into a ZIP archive of much smaller size. We use this for our official model releases, because it's a hassle to distribute model weights that are 2GB+ in size. When using the `--compress` option, it is recommended to specify a test set in order to verify that compression indeed has minimal impact on parsing accuracy. Using the development data for verification is not recommended, since the development data was already used for the model selection criterion during training.
```
python src/export.py export \
  --model-path models/en_bert_base_dev=*.pt \
  --output-dir=models/en_bert_base \
  --test-path=data/wsj/test_23.LDC99T42
```

The `src/export.py` script also has a `test` subcommand that's roughly similar to `python src/main.py test`, except that it supports exported models and has slightly different flags. We can run the following command to verify that our English parser using BERT-large-uncased indeed achieves 95.55 F1 on the canonical WSJ test set:
```
python src/export.py test --model-path benepar_en3_wsj --test-path data/wsj/test_23.LDC99T42
```

## Reproducing Experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for instructions on how to reproduce experiments reported in our ACL 2018 and 2019 papers.

## Citation

If you use this software for research, please cite our papers as follows:

```
@inproceedings{kitaev-etal-2019-multilingual,
    title = "Multilingual Constituency Parsing with Self-Attention and Pre-Training",
    author = "Kitaev, Nikita  and
      Cao, Steven  and
      Klein, Dan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1340",
    doi = "10.18653/v1/P19-1340",
    pages = "3499--3505",
}

@inproceedings{kitaev-klein-2018-constituency,
    title = "Constituency Parsing with a Self-Attentive Encoder",
    author = "Kitaev, Nikita  and
      Klein, Dan",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1249",
    doi = "10.18653/v1/P18-1249",
    pages = "2676--2686",
}
```

## Credits

The code in this repository and portions of this README are based on https://github.com/mitchellstern/minimal-span-parser
