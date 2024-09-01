# FIANA: Financial-News-Analyzer with Transfer Learning
Apply pre-trained models to help quickly grasp and analyze the information from investment news, including three tasks, 1. summarizationm 2. sentiment analysis 3. domain classification

slides: https://docs.google.com/presentation/d/1MrggsmGkW7pWa5fhywEjvF4IK-M2pjVGCfRpWrPOH2s/edit?fbclid=IwAR1VIfE13mFQsOCXklVwMnRCOgwECeWPHaLH2ogaXnaDkQ9zlKL1_ZU-pD8#slide=id.g1c267dd6f93_2_268 

Here is the detail: https://www.dropbox.com/scl/fi/8ubber3p4m3etutcaj7vr/Financial-News.pdf?rlkey=72f4olz8mk55ypr44l3o98yv6&dl=1
## To run this code
### requirements
```shell
torch==1.12.1
transformers==4.22.2
datasets
accelerate
sentencepiece
rouge
spacy
nltk
ckiptagger
tqdm
pandas
numpy
jsonlines
evaluate
rouge_score
opencc

```
## 1. Summary Generation with T5 model



### Download file required and environment settings
```shell
bash scripts/download.sh
```

### Train
you could see the code details in train_summary.py.
Evaluation is included.

```shell
bash scripts/train.sh
```
### Evaluation
if you need only evaluation without training, use this script.
```shell
bash scripts/eval.sh
```
### Evaluation with different decoder algs
```shell
## e.g. bash strategies/beam/beam6/eval.sh
bash strategies/{strategy}/eval.sh
```

### Inference
only to inference(predict)
```shell
bash scripts/predict.sh ./path/to/test_file ./path/to/output_file
```

## 2. Sentiment Analysis
Need more correct data to enhance the result, this is still need more revision(dataset correctness), you could check that in our paper.
### Download file required and environment settings
```shell
bash scripts/download.sh
```
### Train with Evaluation
```shell
bash scripts/train.sh
```

## 3. Domain Classification
Need more correct data to enhance the result, this is still need more revision(for multi-label issues), you could check that in our paper.

### Download file required and environment settings
```shell
bash scripts/download.sh
```
### Train with Evaluation
```shell
bash scripts/train.sh
```

### Inference
only to inference(predict), this task doesn't provide model weight, need to train first to do inference here.
```shell
bash scripts/predict.sh ./path/to/test_file ./path/to/output_file
```
