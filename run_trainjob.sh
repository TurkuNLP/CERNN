#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 06:15:00
#SBATCH -J lm
#SBATCH -o lm.out
#SBATCH -e lm.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

source $USERAPPL/activate_gpu.sh
# run your script
zcat /wrk/jmnybl/parsebank_v4_UD.conllu.gz | THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py
deactivate

# sbatch run_trainjob.sh
