#!/bin/bash
#
#SBATCH --job-name=m_niflow_job
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal,parietal
#SBATCH --mem-per-cpu=20
#SBATCH --error error_%A_%a.out
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00

#
#SBATCH --array=45-47

srun python m_niflow_multiproc_lite.py --config mg_test.cfg --id_to_process $SLURM_ARRAY_TASK_ID
