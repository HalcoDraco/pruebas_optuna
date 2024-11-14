#!/bin/bash

#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --qos=gp_training
#SBATCH --job-name=optuna_test_cpu
#SBATCH --time=24:00:00
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
tasks=56
ntrialpertask=$((ntrials/tasks+1))
resto=$((ntrials%tasks))

for i in $(seq 1 $tasks); do
    if [ $resto -eq 0 ]
    then
        ntrialpertask=$((ntrialpertask-1))
    fi
    if [ $ntrialpertask -eq 0 ]
    then
        break
    fi
    timeout 500 singularity run --app launch_optuna_file optuna.sif notebook_extract.py $ntrialpertask &
    resto=$((resto-1))
    sleep 2
done

wait

cp $WORK_DIR/*.sqlite3 $HOME_DIR/

echo "Todos los procesos han terminado"
