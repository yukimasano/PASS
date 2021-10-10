#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=PASSify-person
#SBATCH --array=0-175%50  # being nice and not running more than 50 jobs in parallel



echo $SLURM_ARRAY_TASK_ID
X=$((${SLURM_ARRAY_TASK_ID}*80000))
Y=$(((${SLURM_ARRAY_TASK_ID} + 1)*80000))

in_file=_tmp_noface_${SLURM_ARRAY_TASK_ID}.txt
rm ${in_file}
results_dir='persondetector_results/'
mkdir -p ${results_dir}
< 1_no_faces.txt tail -n +"$X" | head -n "$((Y - X))" >> ${in_file}

echo "from " ${X}
echo "to " ${Y}

# ETA 4-6Hz
/scratch/shared/beegfs/yuki/envs/py37/bin/python3 -W ignore main_person_detector.py \
  --img_list=${in_file} \
  --save_folder=${results_dir}