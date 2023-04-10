# BNA Parser
![BNA_overall_img (7)](https://user-images.githubusercontent.com/59141702/230888797-75077cda-e321-4179-a9ba-5a3f01e005cd.png)
## Requirements

* python 3.7 or higher
* To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

## Training

* For Penn TreeBank (PTB)

without pre-trained model
```
python src/main.py train --num-layers 8 --model-path-base models/BNA --use-encoder --use-chars-lstm --use-bdmsa --use-nsa --batch-size 250 --learning-rate 0.0008
```
with pre-trained model
```
python src/main.py train --use-pretrained --model-path-base models/BNA_xlnet --use-encoder --use-bdmsa --use-nsa
```

* For Chinese Penn TreeBank (CTB)

without pre-trained model
```
python src/main.py train --num-layers 8 --train-path "data/ctb_5.1/ctb.train" --dev-path "data/ctb_5.1/ctb.dev" --test-path "data/ctb_5.1/ctb.test" --text-processing "chinese" --use-chars-lstm --model-path-base models/BNA_chinese --ngram 3  --batch-size 250 --learning-rate 0.0008 --residual-drop 0.1 --morpho-emb-dropout 0.2 --attention-dropout 0.1 --relu-dropout 0.1 --use-tags --use-encoder --use-bdmsa --use-nsa
```
with pre-trained model
```
python src/main.py train --train-path "data/ctb_5.1/ctb.train" --dev-path "data/ctb_5.1/ctb.dev" --test-path "data/ctb_5.1/ctb.test" --text-processing "chinese" --use-pretrained --pretrained-model "bert-base-chinese" --model-path-base models/BNA_bert_chinese --learning-rate 3e-5 --ngram 3 --batch-size 50 --residual-drop 0.1 --morpho-emb-dropout 0.2 --attention-dropout 0.1 --relu-dropout 0.1 --use-tags --use-encoder --use-bdmsa --use-nsa
```

### Test best-performing models

* PTB
```
python src/main.py test --model-path models/BNA_xlnet
```
* CTB
```
python src/main.py test --test-path "data/ctb_5.1/ctb.test" --text-processing "chinese" --model-path models/BNA_bert_chinese
```

