#################################################
# Example for call LTP libraries under UNIX     #
#################################################
cc=g++
ccflags=-O2
all: cws\
	pos \
	par \
	ner \
	srl 

cws: cws.cpp
	${cc} ${ccflags} -o cws cws.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-L../lib/ -lsegmentor -lboost_regex

pos: pos.cpp
	${cc} ${ccflags} -o pos pos.cpp -I./ \
		-I../include/ \
		-L../lib/ -lpostagger

ner: ner.cpp
	${cc} ${ccflags} -o ner ner.cpp -I./ \
		-I../include \
		-L../lib/ -lner

par: par.cpp
	${cc} ${ccflags} -o par par.cpp -I./ \
		-I../include \
		-L../lib -lparser

srl: srl.cpp
	${cc} ${ccflags} -o srl srl.cpp -I./ \
		-I../include \
		-L../lib -lsrl

.PHONY: clean

clean:
	rm cws
	rm pos
	rm ner
	rm par
	rm srl
