import csv
import json
import numpy as np
import matplotlib.pyplot as plt

path = "/truba/home/ionur/BHRL/work_dirs/vot/BHRL/first_images_seperate/bike1/20230730_130210.log.json"
seq = "bike1"
losses = []
indexes = np.arange(300)

with open(path, 'r') as file:
  csvreader = csv.reader(file)
  for idx,row in enumerate(csvreader):
    if idx:
      loss_val = row[11].replace(" ", "")
      loss_val = "{" + loss_val + "}"
      json_object = json.loads(loss_val)
      losses.append(float(json_object["loss"]))
plt.plot(indexes, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fine-tune on the first appearance of {}". format(seq))
plt.savefig('/truba/home/ionur/BHRL/work_dirs/vot/BHRL/first_images_seperate/{}/{}.png'.format(seq, seq))