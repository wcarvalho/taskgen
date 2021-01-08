conda env create -f gpu.yaml

source activate sfgen

git clone https://github.com/maximecb/gym-minigrid.git _gym-minigrid
cd _gym-minigrid
pip install --editable .
cd ..

git clone https://github.com/mila-iqia/babyai.git _babyai
cd _babyai
pip install --editable .
cd ..

git clone https://github.com/astooke/rlpyt _rlpyt
cd _rlpyt
pip install --editable .
cd ..


# conda remove --name sfgen --all