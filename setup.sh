conda env create -f gpu.yaml

source activate sfgen

bash install.sh

# conda remove --name sfgen --all