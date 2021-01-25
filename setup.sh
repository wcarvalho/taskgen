conda env create -f gpu.yaml

source activate sfgen

bash install.sh
bash setup_lab.sh

# conda remove --name sfgen --all