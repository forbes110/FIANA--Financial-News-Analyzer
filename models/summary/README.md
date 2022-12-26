## Financial News Summary Generation with T5 model
    this is summary task with T5 model for financial corpus, only for chinese language.
    在中文財經文本上訓練的summary T5 model
read the report here: https://docs.google.com/document/d/1MjdgkyRjzXmGEukVv7WHpbQG32B72nX_FDxfn5nicFk/edit?fbclid=IwAR2p-qPo7D26mZx96BzmDrDvfrR-MjR1vNgZE0NNGtnuNSLhrji9IrJbec4#
read the slides here:
https://docs.google.com/presentation/d/1mFT1yCRMp5oZPPCGzji6Gr_zp60pLIAbSbuLpKT_IMs/edit?fbclid=IwAR0Jl9db8Cv8GD5NfXsQ6ECDR4SPewVdxqBwL_GczlIHmdXVQGgre3c6kEA#slide=id.g180ad59962f_5_561


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
```


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