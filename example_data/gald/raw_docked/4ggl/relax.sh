#!/bin/bash
#SBATCH -J relax
#SBATCH -p normal.q
#SBATCH --exclude=star001,star002,star003,star004,star005,star006,star007,star008,star009,star010,star011,star012,star013,star014,star015,star016,star017,star018,star019,star020,star023,star026,star027,star030,star031,star032,star033,star034,star035,star036,star037,star038,star039,star040,star041,star042,star043,star044,star045,star046,star047,star048,star049
#SBATCH -c 1
#SBATCH -o ./log
#SBATCH -e ./log_error
#SBATCH --nice=100000

source /home/hpark/programs/pyrosetta/setup.sh
cd /home/bbh9955/DfAF_git/data/train/raw_docked/model_dock/4ggl ; /home/bbh9955/anaconda3/envs/pyrosetta/bin/python3.8 /home/bbh9955/DfAF_git/src/data_preprocessing/relaxation/fast_relax.py holo.pdb