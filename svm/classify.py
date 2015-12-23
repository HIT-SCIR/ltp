import os
from time import gmtime, strftime


exe ="./svm-light-TK-1.2.1/svm_classify"
test_file="./data/feature.txt"
model_file="./svm_model/1.model"
output_file="./output/"+test_file.split("/")[-1]+"_" + strftime("%H:%M:%S_%d_%b", gmtime())+".result"

cmd = exe +" "+ test_file+" " + model_file +" "+output_file
print cmd

os.system(cmd)
