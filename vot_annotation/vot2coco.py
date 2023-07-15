from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import os
import json
import copy
import math

cls_name = "volkswagen"
class_id = 7
main_img_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT/{}_imgs/". format(cls_name)
#main_img_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT/person7_imgs_nan_experiment/"
gt_path  = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT/{}/groundtruth.txt". format(cls_name)
#gt_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT/person7_gt_nan_experiment/groundtruth_2.txt"
save_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/{}/vot_{}_test.json". format(cls_name, cls_name)


img_paths = []
bboxes    = []
with open(gt_path, "r") as gt:
    lines = [line for line in gt]

for img_id, line in enumerate(lines):
    #if not img_id or img_id == 694 or img_id == 695 or img_id == 696: #or "nan" in line:
    #if not img_id or "nan" in line:
    if not "nan" in line:
        bboxes.append(line)
        img_path = os.path.join(main_img_path, str(img_id+1).zfill(8) + ".jpg")
        img_paths.append(img_path)


json_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOCdevkit/voc_annotation/voc_test.json"
json_file = open(json_path, "r")
voc_data = json.load(json_file)
copy_voc_data = copy.deepcopy(voc_data)



for idx, data in enumerate(copy_voc_data["images"]):
    data["file_name"] = img_paths[idx]

if len(img_paths) > len(copy_voc_data):
    remain_img_paths = img_paths[len(copy_voc_data["images"]) : ]
    for path in remain_img_paths:
        data_dict = {'file_name': path, 
            'height': int, 
            'width': int, 
            'id': int}
        ann_dict = {'segmentation': [], 
                    'area': int, 
                    'iscrowd': int, 
                    'image_id': int, 
                    'bbox': [], 
                    'category_id': int, 
                    'id': int, 'ignore': int}
        copy_voc_data["images"].append(data_dict)
        copy_voc_data["annotations"].append(ann_dict)
        
for idx, data in enumerate(copy_voc_data["images"]):
    width, height = Image.open(img_paths[idx]).size
    data["width"] = width
    data["height"] = height
    data["id"]     = int(os.path.split(img_paths[idx])[-1].split(".")[0])

copy_voc_data["annotations"] = copy_voc_data["annotations"][:len(img_paths)]

for idx, data in enumerate(copy_voc_data["annotations"]):
    if not math.isnan(float(bboxes[idx].split(",")[1])):
        bbox                   = bboxes[idx].split(",")
        data["ignore"]         = 0
    else:
        bbox                   = [0,0,0,0]
        data["ignore"]         = 1
    data["image_id"]       = int(os.path.split(img_paths[idx])[-1].split(".")[0])
    data["bbox"]           = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
    data["category_id"]    = class_id
    data["id"]             = idx+1
    data["area"]           = int(float(bbox[2])) * int(float(bbox[3]))


print(len(copy_voc_data["images"]))
print(len(copy_voc_data["annotations"]))

print(copy_voc_data["annotations"][0])

print(copy_voc_data["annotations"][-1])

json_object = json.dumps(copy_voc_data, indent=4)

with open(save_path, "w") as outfile:
    outfile.write(json_object)
