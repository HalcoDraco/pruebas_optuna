#!/bin/bash

#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --qos=acc_training
#SBATCH --job-name=optuna_test_gpu
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --account=nct_308

module purge
module load singularity

CONTAINER_IMAGE=optuna.sif

HOME_DIR=/gpfs/home/nct/nct01162/optuna
WORK_DIR=/gpfs/scratch/nct_308/nct01162/optuna

mkdir -p $WORK_DIR

if [ ! -d $WORK_DIR/$CONTAINER_IMAGE ]; then
    cp $HOME_DIR/$CONTAINER_IMAGE $WORK_DIR/ ;
fi
cp $HOME_DIR/notebook_extract.py $WORK_DIR/ ;
cp $HOME_DIR/utils.py $WORK_DIR/ ;
cp -r $HOME_DIR/data $WORK_DIR/ ;

cd $WORK_DIR

ntrials=$1
gpus=4
ntrialpergpu=$((ntrials/gpus+1))
resto=$((ntrials%gpus))

for i in $(seq 1 $gpus); do
    if [ $resto -eq 0 ]
    then
        ntrialpergpu=$((ntrialpergpu-1))
    fi
    if [ $ntrialpergpu -eq 0 ]
    then
        break
    fi
    export CUDA_VISIBLE_DEVICES=$((i-1))
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$((i-1))
    singularity run --nv --app launch_optuna_file optuna.sif notebook_extract.py $ntrialpergpu &
    resto=$((resto-1))
    sleep 2
done

wait

cp $WORK_DIR/*.sqlite3 $HOME_DIR/
cp $WORK_DIR/gpu_utilization_* $HOME_DIR/plots/

echo "Todos los procesos han terminado"