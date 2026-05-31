#!/bin/bash
#SBATCH --job-name=my_cuda_job
#SBATCH --partition=gpu
#SBATCH --account=smearlab
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --cpus-per-task=8
#SBATCH --output=./out_Cuda410_%j
#SBATCH --error=./errors_Cuda410_%j

# Load the modules instead of conda
module load gcc
module load cuda/13.0

# Compile the .cu file with nvcc
nvcc -O2 -o my_program /projects/smearlab/trevorro/gputesting/CS410-Diffusion/stencil.cu && ./my_program
echo "exit: $?"

