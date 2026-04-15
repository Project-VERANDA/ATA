#!/bin/bash
#SBATCH --job-name=modeltraining      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --gres=shard:1           # Generic resources; 1 GPU
#SBATCH --time=01:00:00        # Set a limit on the total run time
#SBATCH --mail-user=luke.flanagan@charite.de
#SBATCH --mail-type=ALL       # Notify user by email in case of job failure
#SBATCH --output=my_job.o%j    # File name for standard output
#SBATCH --error=my_job.e%j     # File name for standard error output

srun --exclusive python spacy_anonymizer/training/finetune.py --params=1 &
srun --exclusive python bert_anonymizer/training/finetune.py --params=2 &
wait
