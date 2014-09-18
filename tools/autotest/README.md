Automatically Test Scripts for LTP
==================================
This set of scripts is used for automatically test ltp's performance,
memory leak, code qualities and some other stuff.

## Prerequisite Packages

Several sofeware packages are required to run this scripts.

- valgrind: for memcheck and callgrind
- cppcheck: for static check
- graphviz: for convert the callgrind into a PNG graph

If some the packages are not available, certain test will be skipped.

## Preparing the data

Please setup the model into the `ltp_data` directory.

## Run it

by 
```
./autotest.py
```

for detailed arguments, please use `./autotest.py -h`

## Output

This test suite will automatically generate the following reports:
- memcheck report for ltp_test
- callgrind and a visuialization for ltp_test call graph
- xml output for the input file (through ltp_test)
- xml output for the input file (through ltp_server)
- speed performance
