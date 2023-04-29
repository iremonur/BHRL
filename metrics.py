

import json

def load_gt(gt_json):
    f = open(gt_json)
    data = json.load(f)
    absent=0
    absent_ids = []
    gt_ids = []
    for ann in data["annotations"][1:]:
        gt_ids.append(ann["image_id"])
        if ann["ignore"]:
            absent+=1  
            absent_ids.append(ann["image_id"]) 
    return absent, absent_ids, gt_ids


# Calculate accuracy percentage between two lists
def accuracy_metric(predictions, gt_json):
    absent, absent_ids, gt_ids = load_gt(gt_json)
    f = open(predictions["bbox"])
    preds = json.load(f)
    tp, fp, fn = 0, 0, 0
    tn = len(gt_ids) - len(preds[1:])
    for pred in preds[1:]:
        if pred["image_id"] in absent_ids:
            fp+=1
    acc = (tp+tn) / (tp+tn+fp+fn)
    return acc

#acc = accuracy_metric()