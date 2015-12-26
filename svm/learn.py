import os

exe ="./svm-light-TK-1.2.1/svm_learn"
train_file="./data/train.txt"
model_file="./svm_model/1.model"
option="-t 5 -C + -r 1"

cmd = exe +" "+option +" "+ train_file+" " + model_file
print cmd

os.system(cmd)

