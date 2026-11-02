#!/bin/bash
# Reproduces our classification runs on any machine with a GPU (Colab, a lab server, a laptop with CUDA).
# It trains, evaluates, and lays the outputs out exactly like the files in results/cls_cnn and results/cls_resnet18.
#
# usage: ./reproduce_cls.sh [cnn|resnet18|all]      (default: all)
# env:   OUT_DIR=<dir>  where to write the results folders (default: ./reproduced, so results/ is never overwritten)
#
# Prerequisites: pip install -r requirements.txt, and the MiniPlaces data in ./data (sh download_cls_dataset.sh).
# Training is not seeded, so accuracies will differ slightly from ours (CNN ~49.8% top-1, ResNet-18 ~57.7%).
set -e
set -o pipefail
WHICH=${1:-all}
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUT_DIR:-$ROOT/reproduced}

if [ ! -f "$ROOT/data/train.txt" ] || [ ! -d "$ROOT/data/images" ]; then
    echo "ERROR: MiniPlaces not found in $ROOT/data. Run: mkdir -p data && sh download_cls_dataset.sh" >&2
    exit 1
fi
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"

run_one() {
    CONFIG=$1      # e.g. cls_miniplaces_cnn.yaml
    DEST=$2        # e.g. cls_cnn
    KEEP_CKPT=$3   # yes/no: copy model_final.pth.tar into the results folder
    NAME=$(basename "$CONFIG" .yaml)
    EXP="$ROOT/logs/${NAME}_${NAME}"   # train_cls.py names the folder <config>_<--output>
    R="$OUT_DIR/$DEST"

    rm -rf "$EXP"
    cd "$ROOT/code"
    python train_cls.py configs/$CONFIG --output $NAME -p 100
    mkdir -p "$R"
    python eval_cls.py configs/$CONFIG "$EXP/model_final.pth.tar" -p 100 2>&1 | tee "$R/eval_$NAME.txt"

    cp configs/$CONFIG "$R/"
    cp "$EXP/config.txt" "$R/"
    rm -rf "$R/tensorboard" && cp -r "$EXP/logs" "$R/tensorboard"
    if [ "$KEEP_CKPT" = "yes" ]; then cp "$EXP/model_final.pth.tar" "$R/"; fi
    echo "=> wrote $R"
}

case $WHICH in
    cnn)      run_one cls_miniplaces_cnn.yaml cls_cnn yes ;;
    resnet18) run_one cls_miniplaces_resnet18.yaml cls_resnet18 no ;;
    all)      run_one cls_miniplaces_cnn.yaml cls_cnn yes
              run_one cls_miniplaces_resnet18.yaml cls_resnet18 no ;;
    *)        echo "usage: $0 [cnn|resnet18|all]" >&2; exit 1 ;;
esac
