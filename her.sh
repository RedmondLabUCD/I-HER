#!/bin/bash -l
#SBATCH --job-name=PushMagicNp2
# specify number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 8

# specify the walltime e.g 20 mins
#SBATCH -t 48:10:00

# set to email at start,end and failed jobs
# SBATCH --mail-type=ALL
# SBATCH --mail-user=robert.mccarthy@ucdconnect.ie
# exclude all except 66 and 67
#SBATCH --exclude=sonic1,sonic2,sonic3,sonic4,sonic5,sonic6,sonic7,sonic8,sonic21,sonic22,sonic23,sonic24,sonic25,sonic26,sonic27,sonic28,sonic29,sonic30,sonic31,sonic32,sonic33,sonic34,sonic35,sonic36,sonic37,sonic38,sonic39,sonic40,sonic43,sonic44,sonic45,sonic46,sonic47,sonic48,sonic49,sonic50,sonic51,sonic52,sonic53,sonic54,sonic55,sonic56,sonic57,sonic58,sonic59,sonic60,sonic61,sonic63,sonic64,sonic65,sonicmem3

# run from current directory
cd $SLURM_SUBMIT_DIR
export FI_PROVIDER=verbs
module load intel/intel-cc
module load intel/intel-mkl
module load intel/intel-mpi

echo "loading in python..."
module load python/3.7.4
echo "Loaded in python"

# change env-name in train.py, not here
mpirun -np 8 python3 -u train.py --env-name='FetchPush-v1' --n-epochs=100 --real-rollouts=512 --dynamics-steps=20000 --dynamics-hiddens=2 --ensemble-size=5 --p-realwreal=0.9 --p-imagwreal=0.1 --bias-recent-refill=1 --bias-recent-sample=2 --bias-real=1 --n-cycles=50 --n-batches=40 --include-ri=1 --distinguish=1 --refill=1 --squared-l2-loss=1 --save-models=1 --seed=0 --exp-name='push_train' 2>&1 | tee push_train.log

