

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
    predicted_ids = []
    tp, fp, fn = 0, 0, 0
    #tn = len(gt_ids) - len(preds[1:])
    for pred in preds[1:]:
        if not int(pred["image_id"]) == 1:
            predicted_ids.append(pred["image_id"])
            if pred["image_id"] in absent_ids:
                fp+=1

    print(gt_ids)
    print("#")
    print(predicted_ids)
    
    print("LEN GT : ", len(gt_ids))
    print("LEN PREDS : ", len(predicted_ids))
    tn = len(set(gt_ids).difference(predicted_ids))
    acc = 100 * (tp+tn) / (tp+tn+fp+fn) 
    print("FP : ", fp)
    print("TN : ", tn)
    return acc

#acc = accuracy_metric(predictions="/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_results_tightrope.pkl.bbox.json", gt_json="/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/person7/vot_person7_test_absent.json")