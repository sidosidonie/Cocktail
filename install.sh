mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -c conda-forge cupy nccl cudatoolkit=11.8
pip install -r requirements.txt
