#!/usr/bin/env bash
#SBATCH --job-name=MD5
#SBATCH --partition=wacc
#SBATCH --time=00-00:0:03
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o out.out -e error.err

module load nvidia/cuda

make

./a.out