import os

path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation"
new_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/work_dirs/vot/BHRL/first_images_seperate"

for root, dirs, files in os.walk(path):
    if not "_imgs" in os.path.split(root)[-1] and not "ft" in os.path.split(root)[-1] and not "gt_" in os.path.split(root)[-1] and not "results_" in os.path.split(root)[-1] and not "vot_annotation" in os.path.split(root)[-1]:
        print(os.path.split(root)[-1])
        if not os.path.exists(os.path.join(new_path, os.path.split(root)[-1])):
            os.mkdir(os.path.join(new_path, os.path.split(root)[-1]))
