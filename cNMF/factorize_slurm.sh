#!/bin/bash
#SBATCH -J cNMF_fact # Job name
#SBATCH --time=3:00:00
#SBATCH --mem=3G

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo ""
echo "task $SLURM_ARRAY_TASK_ID of $SLURM_NTASKS"
echo ""

$1 factorize --output-dir $2 --name $3 --worker-index $SLURM_ARRAY_TASK_ID
