#!/bin/bash

#SBATCH -p TeslaV100
#SBATCH -J autokeras 
#SBATCH --output=autokeras_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6

#srun singularity exec --nv /home/albert/torch_albert.simg python retrain.py

 srun singularity exec --nv /home/albert/tf_latest_auto_keras.simg python prise_en_main_autokeras.py
# srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/albert/Containers/torch_albert.simg python retrain.py --arc-checkpoint ./checkpoint.json

# srun singularity exec --nv /home/albert/torch_albert.simg python dartsFastImplement.py

