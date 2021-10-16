if [ -z $1 ]; then # empty
    arch=gpu
else
    arch="$1"
fi

conda env create -f projects/thor_nav/$arch.yaml

eval "$(conda shell.bash hook)"

conda activate thor_nav

git clone https://github.com/wcarvalho/gym-minigrid.git _gym-minigrid
cd _gym-minigrid
git checkout wilka
pip install --editable .
cd ..

git clone https://github.com/wcarvalho/babyai.git _babyai
cd _babyai
git checkout wilka
pip install --editable .
cd ..

git clone https://github.com/wcarvalho/rlpyt _rlpyt
cd _rlpyt
git checkout wilka
pip install --editable .
cd ..


#setup lab
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager


# conda remove --name thor_nav --all
# conda env update --file gpu.yaml