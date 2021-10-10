import glob
import sys

step = int(sys.argv[1])

if step == 0:
    dir = str(sys.argv[2])
    # enlist all files
    files = glob.glob(dir + '/*/**')
    f = open("0_all_files.txt", "a")
    i = 0
    for file in files:
        i += 1
        f.write(file +"\n")
    f.close()
    print(f"found {i} files")
    nj = (i // 80000) +1 # number of slurm jobs
    f = open('face/sbatch_face.sh','a')
    command = f"""#!/bin/bash
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=compute                   # we only need CPUs, adapt to your cluster
#SBATCH --array=0-{nj}%50                     # being nice and not running more than 50 jobs in parallel
#SBATCH --job-name=PASSify-face
#SBATCH --constraint=10GbE


cd face/
echo $SLURM_ARRAY_TASK_ID
X=$(($SLURM_ARRAY_TASK_ID*80000))
Y=$((($SLURM_ARRAY_TASK_ID + 1)*80000))

in_file=_tmp_all_files_$SLURM_ARRAY_TASK_ID.txt
rm $in_file
results_dir='/facedetector_results/'
< all_files.txt tail -n +"$X" | head -n "$((Y - X))" >> $in_file

echo "from " $X
echo "to " $Y


# ETA 4-6Hz
python3 -W ignore main_face_detector.py \
  --input_txt=$in_file \
  --save_folder=$results_dir \
  --cpu
"""
    f.write(command)
    f.close()


if step == 1:
    files = glob.glob('facedetector_results/noface_index/*/**')
    f = open("1_no_faces.txt", "a")
    i = 0
    for file in files:
        i += 1
        f.write(file +"\n")
    f.close()
    print(f"left with {i} images that do not contain faces")
    nj = (i // 80000) +1 # number of slurm jobs
    f = open('person/sbatch_person.sh','a')
    command = f"""#!/bin/bash
#SBATCH --mem=10G
#SBATCH --cpus-per-task=5
#SBATCH --time=8:00:00                          # this is on the low-end, jobs might finish quicker.
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu                         # we need GPUs, adapt to your cluster
#SBATCH --job-name=PASSify-person
#SBATCH --array=0-{nj}%50                       # being nice and not running more than 50 jobs in parallel



echo $SLURM_ARRAY_TASK_ID
X=$(($SLURM_ARRAY_TASK_ID*80000))
Y=$((($SLURM_ARRAY_TASK_ID + 1)*80000))

in_file=_tmp_noface_$SLURM_ARRAY_TASK_ID.txt
rm $in_file
results_dir='persondetector_results/'
mkdir -p $results_dir
< 1_no_faces.txt tail -n +"$X" | head -n "$((Y - X))" >> $in_file

echo "from " $X
echo "to " $Y

# ETA 4-6Hz
python3 -W -W ignore main_person_detector.py \
  --img_list=$in_file \
  --save_folder=$results_dir 
"""
    f.write(command)
    f.close()

if step == 2:
    files = glob.glob('persondetector_results/noperson_index/*/**')
    f = open("2_no_faces__no_person.txt", "a")
    i = 0
    for file in files:
        i += 1
        f.write(file +"\n")
    f.close()
    print(f"left with {i} images that do not contain faces nor persons")
    print("final file of images left can be found in: 2_no_faces__no_person.txt")
    print("Note that the results are from automated algorithms,"
          " and thus do not work 100% well and might work differently well on different humans, possibly introducing bias. "
          "For all real applications, please thoroughly run human evaluations.")
