import os

commondir= os.getcwd()+"/"
exe="./bin/ltp_test"

option={}
option["threads"] = "4"
option["input"] = commondir + "data/input.txt"
'''
option["segmentor-model"] = commondir +"ltp_data/cws.model"
option["postagger-model"] = commondir +"ltp_data/pos.model"
option["ner-model"] = commondir +"ltp_data/ner.model"
option["parser-model"] = commondir +"ltp_data/parser.model"
option["srl-data"] = commondir +"ltp_data/srl_data/"
'''
cmd= exe;
for k in option:
	cmd= cmd + " --"+ k+"="+option[k]

print cmd
os.system(cmd)

