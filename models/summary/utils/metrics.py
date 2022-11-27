import os
from ckiptagger import WS, data_utils
from rouge import Rouge
import zipfile, gdown

cache_dir = os.environ.get("XDG_CACHE_HOME", "~/.cache")
download_dir = os.path.join(cache_dir, "ckiptagger")
data_dir = os.path.join(cache_dir, "ckiptagger/data")

def download_data_gdown(path):
    url = "https://www.dropbox.com/s/3s1m0lbxu45cctf/data.zip?dl=1"
    data_zip = os.path.join(path, "data.zip")
    gdown.download(url, data_zip, quiet=False)
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    return

os.makedirs(download_dir, exist_ok=True)
if not os.path.exists(os.path.join(data_dir, "model_ws")):
    download_data_gdown(download_dir)

ws = WS(data_dir)

def tokenize_and_join(sentences):
    return [" ".join(toks) for toks in ws(sentences)]

rouge = Rouge()

def get_rouge(preds, refs, avg=True, ignore_empty=False):
    """wrapper around: from rouge import Rouge
    Args:
        preds: string or list of strings
        refs: string or list of strings
        avg: bool, return the average metrics if set to True
        ignore_empty: bool, ignore empty pairs if set to True
    """
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(refs, list):
        refs = [refs]
    preds, refs = tokenize_and_join(preds), tokenize_and_join(refs)
    return rouge.get_scores(preds, refs, avg=avg, ignore_empty=ignore_empty)