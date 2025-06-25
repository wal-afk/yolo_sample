#%%
from glob import glob
import csv
from pprint import pprint

dir="dataset/sample1"

def count_label(label_list, dir_path):
    label_count={}
    for path in glob(f"{dir_path}/*.txt"):
        with open(path) as f:
            reader=csv.reader(f,delimiter=" ")
            for line in reader:
                label=label_list[int(line[0])]
                if label not in label_count:
                    label_count[label]=0
                label_count[label]+=1
    return label_count

label_list=[]
with open(f"{dir}/labels.txt") as f:
    for line in f.readlines():
        label_list.append(line.strip())

label_count_train=count_label(label_list,f"{dir}/labels/train")

pprint(label_count_train)
print("sum=", sum(label_count_train.values()))

# %%
