## install packages
pip install -r requirements.txt

git clone https://github.com/huggingface/transformers.git
git checkout t5-fp16-no-nans
pip install -e .

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