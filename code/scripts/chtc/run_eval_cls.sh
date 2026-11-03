#!/bin/bash
# Evaluates one classification checkpoint on the MiniPlaces validation set (CHTC Docker job).
# usage: run_eval_cls.sh <config yaml name, e.g. cls_miniplaces_cnn.yaml> <checkpoint path>
set -e
set -o pipefail
echo "args: [$@]"
CONFIG=$1
CKPT=$(basename "$2")   # the checkpoint is transferred into the job's top-level directory
if [ -z "$CONFIG" ] || [ -z "$CKPT" ]; then echo "ERROR: need <config> <checkpoint>" >&2; exit 1; fi
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint $CKPT not found" >&2; ls; exit 1; fi
NAME=$(basename "$CONFIG" .yaml)

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
python eval_cls.py configs/$CONFIG ../$CKPT -p 100 2>&1 | tee ../eval_$NAME.txt
