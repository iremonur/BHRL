import json
import copy
import os

seq_cls = "yamaha"
path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/{}/vot_{}_test.json". format(seq_cls, seq_cls)
save_path = "/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/ft/{}_first_ft.json". format(seq_cls)

json_file = open(path, "r")
data = json.load(json_file)

for key in data.keys():
    print(key)

print(len(data["images"]))
print(len(data["type"]))
print(len(data["annotations"]))
print(len(data["categories"]))

# LENGTH = int(len(data["images"]) / 4)
# print(LENGTH)
LENGTH = 1

copy_data = copy.deepcopy(data)

# copy_data["annotations"] = data["annotations"][:1]
# copy_data["images"] = data["images"][:1]

# cnt = 2
# for root, dirs, files in os.walk("/home/ionur2/Desktop/MSc_THESIS/BHRL/vot_annotation/", topdown=False):
#     for file in files:
#         if (file.endswith(".json")) and (not "absent" in file) and (not "ballet_" in file) and (not "bull_" in file) and (not "deer_" in file) and (not "nissan_" in file) and (not "volkswagen_" in file) and (not "vot_test" in file):
#             json_file = open(os.path.join(root,file), "r")
#             data = json.load(json_file)

#             data["annotations"][:1][0]["id"] = cnt
#             data["annotations"][:1][0]["image_id"] = cnt
#             copy_data["annotations"].append(data["annotations"][:1][0])

#             data["images"][:1][0]["id"] = cnt
#             copy_data["images"].append(data["images"][:1][0])

#             cnt+=1

copy_data["annotations"] = data["annotations"][:LENGTH]
copy_data["images"] = data["images"][:LENGTH]

json_object = json.dumps(copy_data, indent=4)

with open(save_path, "w") as outfile:
    outfile.write(json_object)

print(len(copy_data["images"]))
print(len(copy_data["type"]))
print(len(copy_data["annotations"]))
print(len(copy_data["categories"]))


