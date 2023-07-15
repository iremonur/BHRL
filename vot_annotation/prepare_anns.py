
from PIL import Image
import os
import json
import copy
import math

seq = "group3"
ABSENT=False
cls_name = "person"
gt_path  = "/home/ionur2/Desktop/MSc_THESIS/BHRL/data/VOT/{}/groundtruth.txt". format(seq.split("_")[0])
gt_save_path = "/home/ionur2/Desktop/MSc_THESIS/mAP/input/{}/ground-truth". format(seq)
res_save_path = "/home/ionur2/Desktop/MSc_THESIS/mAP/input/{}/detection-results". format(seq)
result_file = "/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_results_{}.pkl.bbox.json". format(seq)



def get_result_ids(result_file):

    f = open(result_file)
    data = json.load(f)
    ids = []
    for frame in data:
        ids.append(int(frame["image_id"]))
    return ids

def get_gt_ids(gt_path):

    ids = []
    nan_ids = []
    with open(gt_path, "r") as gt:
        lines = [line for line in gt]

    for idx, line in enumerate(lines):
        coor = line.split(",")
        if "nan" in coor[1]:
            nan_ids.append(idx+1)
        else:
            ids.append(idx+1)
    return ids, nan_ids

def get_fn_ids(predicted_ids, gt_ids):

    fn_ids = set(gt_ids).difference(predicted_ids)
    return fn_ids

def prep_gt(cls_name, gt_path, save_path):

    with open(gt_path, "r") as gt:
        lines = [line for line in gt]
    c = 0
    for idx, line in enumerate(lines):
        coor = line.split(",")
        if not "nan" in coor[1]:
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[0]) + float(coor[2])
            y2 = float(coor[1]) + float(coor[3])
            new_line = cls_name + " " +str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
        else:
            continue
            #new_line = "{} 0 0 0 0". format(cls_name)
        with open(os.path.join(save_path, "{}.txt". format(str(idx+1).zfill(8))), 'w') as f:
            f.write(new_line)



def prep_res(cls_name, result_file, save_path, fn_ids):
    f = open(result_file)
    data = json.load(f)
    for idx, frame in enumerate(data):
        id = str(frame["image_id"]).zfill(8)
        coor = frame["bbox"]
        x1 = float(coor[0])
        y1 = float(coor[1])
        x2 = float(coor[0]) + float(coor[2])
        y2 = float(coor[1]) + float(coor[3])
        score = str(frame["score"])
        with open(os.path.join(save_path, id+".txt"), 'w') as f:
            f.write(cls_name + " " + score + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
        if not ABSENT:
            for fn_id in fn_ids:
                with open(os.path.join(save_path, str(fn_id).zfill(8)+".txt"), 'w') as f:
                    f.write(cls_name + " " + "0" + " " + "0"+ " " + "0" + " " + "0" + " " + "0")

"""
prep_gt(cls_name, gt_path, gt_save_path)

res_ids = get_result_ids(result_file)
gt_ids, nan_ids = get_gt_ids(gt_path)
fn_ids = get_fn_ids(res_ids, gt_ids)
print(fn_ids)
prep_res(cls_name, result_file, res_save_path, fn_ids)
"""
with open("/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/group3/vot_group3_test.json", 'r') as f:
    dataset = json.load(f)
print(len(dataset["images"]))