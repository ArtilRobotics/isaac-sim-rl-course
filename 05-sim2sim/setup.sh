#!/usr/bin/env bash
# =============================================================================
# Setup del entorno para sim2sim (Isaac Lab → MuJoCo)
#
# Crea un entorno conda separado del de isaaclab,
# instala mujoco y las dependencias mínimas.
#
# Uso:
#   source ~/miniconda3/etc/profile.d/conda.sh   # activar conda
#   bash setup.sh
# =============================================================================

set -euo pipefail

ENV_NAME="sim2sim"
PYTHON_VERSION="3.11"

echo ""
echo "============================================="
echo "  Setup: entorno conda '${ENV_NAME}'"
echo "============================================="

# ── Crear entorno conda ────────────────────────────────────────────────────────
echo ""
echo "[setup] Creando entorno conda '${ENV_NAME}' con Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo ""
echo "[setup] Activando entorno..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Instalar dependencias ──────────────────────────────────────────────────────
echo ""
echo "[setup] Instalando dependencias..."
pip install --upgrade pip

pip install \
    mujoco \
    torch  \
    scipy  \
    numpy

echo ""
echo "[setup] Versiones instaladas:"
python -c "import mujoco; print(f'  mujoco  : {mujoco.__version__}')"
python -c "import torch;  print(f'  torch   : {torch.__version__}')"
python -c "import scipy;  print(f'  scipy   : {scipy.__version__}')"
python -c "import numpy;  print(f'  numpy   : {numpy.__version__}')"

# ── Mensaje final ──────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  ✅ Setup completado!"
echo "============================================="
echo ""
echo "Próximos pasos:"
echo ""
echo "  1. Activar el entorno:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. Ejecutar sim2sim:"
echo "     python sim2sim.py"
echo ""
echo "  3. O en modo headless (sin ventana):"
echo "     python sim2sim.py --no_render --num_episodes 20"
echo ""
