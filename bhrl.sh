#!/bin/bash
#SBATCH -p akya-cuda        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ionur       # Kullanici adi
#SBATCH -J car16       # Gonderilen isin ismi
#SBATCH -o /truba/home/ionur/BHRL/target_update_studies/car16_deneme.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=00:40:00      # Sure siniri koyun.

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate BHRL
python tools/test.py --config configs/vot/BHRL.py --seq_name car16 --checkpoint work_dirs/vot/BHRL/first_images_seperate/car16/epoch_100.pth --eval bbox