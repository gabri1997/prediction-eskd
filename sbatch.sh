#!/bin/bash
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --output=res_ESKD_NN.out
#SBATCH --error=res_ESKD_NN.err
#SBATCH --job-name=ESKD_NN
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Info sul job
echo "🔹 Job iniziato su $(hostname)"
echo "🔹 Data e ora: $(date)"
echo "🔹 GPU disponibile: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'Nessuna GPU disponibile!')"

. /usr/local/anaconda3/etc/profile.d/conda.sh


# Lancio il training script
python /work/grana_far2023_fomo/ESKD/net.py

echo "✅ Job completato con successo!"
echo "🔹 Data e ora di fine: $(date)"
echo "🔹 GPU utilizzata: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'Nessuna GPU disponibile!')"
