#!/bin/bash
# Runs one classification training job inside the CHTC Docker container.
# usage: run_cls.sh <config yaml name, e.g. cls_miniplaces_cnn.yaml> <run name>
set -e
CONFIG=$1
NAME=$2

nvidia-smi
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUSERBASE=$PWD/.pyuser
export PATH=$PYTHONUSERBASE/bin:$PATH
# torchvision 0.19.1 matches torch 2.4.1 in the image
pip install --user -q torchvision==0.19.1 gdown pyyaml tensorboard opencv-python-headless pillow
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# dataset (same steps as download_cls_dataset.sh, without needing wget)
mkdir -p data && cd data
gdown 16GYHdSWS3iMYwMPv5FpeDZN2rH7PR0F2
python - <<'PY'
import urllib.request
base = "https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/"
for f in ("train.txt", "val.txt"):
    urllib.request.urlretrieve(base + f, f)
PY
tar -xzf data.tar.gz && rm data.tar.gz
cd ..

cd code
python train_cls.py configs/$CONFIG --output $NAME -p 100
cd ..

# only ship back logs / checkpoints / tensorboard events
tar -czf logs_$NAME.tar.gz logs
