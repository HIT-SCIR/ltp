import os

exe ="./svm-light-TK-1.2.1/svm_learn"
train_file="./demo_data/arg0.train"
model_file="./svm_model/1.model"
option="-t 5"

cmd = exe +" "+option +" "+ train_file+" " + model_file
print cmd

os.system(cmd)

