## install packages
pip install -r requirements.txt

## get_rouge
# git clone https://github.com/moooooser999/ADL22-HW3.git
# cd ADL22-HW3
# pip install -e tw_rouge
# cd ..

## tw_rouge need fixed package
# wget https://www.dropbox.com/s/0fqay35uddin2vv/data.zip?dl=1 -O ./data.zip
# cp ./data.zip /root/.cache/ckiptagger

## fix fp16
git clone https://github.com/huggingface/transformers.git
git checkout t5-fp16-no-nans
pip install -e .

## download dataset
# if [ ! -f ./data/train.jsonl ]; then
#     wget https://www.dropbox.com/s/1w01skdqokgoqm0/train.jsonl?dl=1 -O ./data/train.jsonl
# fi

# if [ ! -f ./data/public.jsonl ]; then
#     wget https://www.dropbox.com/s/jkyn10uyb5ihj7l/public.jsonl?dl=1 -O ./data/public.jsonl
# fi

## tokenizer
if [ ! -f ./ckpt/tokenizer.json ]; then
    wget https://www.dropbox.com/s/xhz413up2u57bow/tokenizer.json?dl=1 -O ./ckpt/tokenizer.json
fi

## model
if [ ! -f ./ckpt/pytorch_model.bin ]; then
    wget https://www.dropbox.com/s/2uh18qe7neyd1di/pytorch_model.bin?dl=1 -O ./ckpt/pytorch_model.bin
fi


# /Users/zhouchengkang/Desktop/WareHouse/Financial-News-Analyzer/models/summary

## data
if [ ! -f ./data/train_file.json ]; then
    wget https://www.dropbox.com/s/92c3t4f90xhsbr9/train_file.json?dl=1 -O ./data/train_file.json
fi

if [ ! -f ./data/valid_file.json ]; then
    wget https://www.dropbox.com/s/yi8555pbfgnquzg/valid_file.json?dl=1 -O ./data/valid_file.json
fi

if [ ! -f ./data/test_file.json ]; then
    wget https://www.dropbox.com/s/2t6o4hbqnkfxpbf/test_file.json?dl=1 -O ./data/test_file.json
fi