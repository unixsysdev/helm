#!/bin/bash
# ==============================================================================
# HELM-D H200 Environment Setup — One-Shot
# ==============================================================================
# Run this ONCE when provisioning a new H200 VM.
# Installs all deps, patches geoopt for torch.compile, clones repo.
#
# Usage: bash setup_h200.sh
# ==============================================================================
set -e

export PATH=/opt/conda/bin:$PATH
echo "============================================================"
echo "HELM-D H200 Setup"
echo "============================================================"

# 1. Core Python deps
echo -e "\n[1/6] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers \
    datasets \
    geoopt \
    tokenizers \
    huggingface_hub \
    accelerate \
    sentencepiece \
    protobuf

# 2. Flash Attention 2 (NVIDIA only, needs CUDA headers)
echo -e "\n[2/6] Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation --quiet 2>/dev/null || {
    echo "  flash-attn build failed, trying pre-built wheel..."
    pip install flash-attn --quiet 2>/dev/null || {
        echo "  WARNING: flash-attn not available, will use SDPA fallback"
    }
}

# 3. Verify critical imports
echo -e "\n[3/6] Verifying imports..."
python3 -c "
import torch
print(f'  torch:        {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0)}')
print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'  transformers: {transformers.__version__}')

import datasets
print(f'  datasets:     {datasets.__version__}')

import geoopt
print(f'  geoopt:       {geoopt.__version__}')

try:
    from flash_attn import flash_attn_func
    print(f'  flash_attn:   OK')
except ImportError:
    print(f'  flash_attn:   NOT AVAILABLE (will use SDPA)')
"

# 4. Patch geoopt for torch.compile compatibility
# torch.norm(p=2) is not traceable by TorchInductor — must use torch.linalg.vector_norm
echo -e "\n[4/6] Patching geoopt for torch.compile..."
GEOOPT_MATH=$(python3 -c "import geoopt; import os; print(os.path.join(os.path.dirname(geoopt.__file__), 'manifolds', 'lorentz', 'math.py'))")
if [ -f "$GEOOPT_MATH" ]; then
    if grep -q "torch.norm" "$GEOOPT_MATH"; then
        sed -i 's/torch\.norm(\([^,]*\), p=2, dim=\([^)]*\))/torch.linalg.vector_norm(\1, ord=2, dim=\2)/g' "$GEOOPT_MATH"
        echo "  Patched: torch.norm -> torch.linalg.vector_norm in $GEOOPT_MATH"
    else
        echo "  Already patched or no torch.norm found"
    fi
else
    echo "  WARNING: geoopt math.py not found at $GEOOPT_MATH"
fi

# 5. Clone/update repo
echo -e "\n[5/6] Setting up HELM-D repo..."
if [ ! -d "/root/helm-src" ]; then
    git clone https://github.com/unixsysdev/helm.git /root/helm-src 2>/dev/null || true
    cd /root/helm-src && git checkout h200-optimizations 2>/dev/null || true
fi

# Symlink so imports work: helm-src/helm -> importable
export PYTHONPATH="/root/helm-src:$PYTHONPATH"
echo "  PYTHONPATH=$PYTHONPATH"

# 6. Pre-download tokenizer (avoid first-run delay)
echo -e "\n[6/6] Pre-downloading tokenizer..."
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print(f'  TinyLlama 32K tokenizer: {len(tok)} tokens')
"

# Summary
echo -e "\n============================================================"
echo "Setup complete. Run training with:"
echo ""
echo "  export PYTHONPATH=/root/helm-src:\$PYTHONPATH"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  nohup python3 -O train_cot.py --save_dir /tmp/checkpoints/cot > train.log 2>&1 &"
echo "============================================================"
