#!/bin/bash
#SBATCH --array=0-X%50 # being nice and not running more than 50 jobs in parallel
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=compute # we only need CPUs
#SBACTH --open-mode=append
#SBATCH --job-name=PASSify-face
#SBATCH --constraint=10GbE


echo $SLURM_ARRAY_TASK_ID
X=$((${SLURM_ARRAY_TASK_ID}*80000))
Y=$(((${SLURM_ARRAY_TASK_ID} + 1)*80000))

in_file=_tmp_all_files_${SLURM_ARRAY_TASK_ID}.txt
rm ${in_file}
results_dir='/facedetector_results/'
< all_files.txt tail -n +"$X" | head -n "$((Y - X))" >> ${in_file}

echo "from " ${X}
echo "to " ${Y}

# ETA 4-6Hz
/scratch/shared/beegfs/yuki/envs/py37/bin/python3 -W ignore main_face_detector.py \
  --input_txt=${in_file} \
  --save_folder=${results_dir} \
  --cpu