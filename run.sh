#!/bin/bash

echo "🚗 Iniciando Sistema de seguimiento de trafico"

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python main.py \
    --config config.yaml \
    --workers 4 \
    --buffer 15 \
    --batch \
    --batch-size 4 \
    --cpu-mode
