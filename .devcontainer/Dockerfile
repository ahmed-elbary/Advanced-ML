FROM tensorflow/tensorflow:2.17.0-gpu
RUN pip install --upgrade pip 

## Install PyTorch and other deps
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu126/torch_stable.html
RUN pip install ipykernel pillow vizdoom stable-baselines3 opencv-python
