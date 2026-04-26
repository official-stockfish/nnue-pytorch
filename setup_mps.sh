# !/bin/bash
# Note: This script only sets up the MPS enviroment for pytorch.
# It does not install system-wide packages like cmake or python-dev
# which are required for data-loader compilation

python3.12 -m venv ./pytorch_mps_env
source ./pytorch_mps_env/bin/activate

pip install --upgrade pip
pip install torch==2.8.0 torchvision torchaudio

pip install --no-cache-dir -r requirements.txt

chmod +x setup_script.sh
./setup_script.sh