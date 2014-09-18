#################################################
# Example for call LTP libraries under UNIX     #
#################################################
cc=g++
ccflags=-O2
all: cws cws_cmdline multi_cws_cmdline \
	pos pos_cmdline multi_pos_cmdline \
	par par_cmdline \
	ner \
	srl 

cws: cws.cpp
	${cc} ${ccflags} -o cws cws.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-L../lib/ -lsegmentor -lboost_regex

cws_cmdline: cws_cmdline.cpp
	${cc} ${ccflags} -o cws_cmdline cws_cmdline.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-Wl,-dn -L../lib/ -lsegmentor -lboost_regex -Wl,-dy

#
# tinythread is directly invoked into the example
#
multi_cws_cmdline: multi_cws_cmdline.cpp
	${cc} ${ccflags}  -o multi_cws_cmdline multi_cws_cmdline.cpp \
		../thirdparty/tinythreadpp/tinythread.cpp \
		-I./ \
		-I../include/ \
		-I../thirdparty/boost/include/ \
		-I../thirdparty/tinythreadpp/ \
		-Wl,-dn -L../lib/ -lsegmentor -lboost_regex -Wl,-dy -lpthread

pos: pos.cpp
	${cc} ${ccflags} -o pos pos.cpp -I./ \
		-I../include/ \
		-L../lib/ -lpostagger

pos_cmdline: pos_cmdline.cpp
	${cc} ${ccflags} -o pos_cmdline pos_cmdline.cpp -I./ \
		-I../include/ \
		-Wl,-dn -L../lib/ -lpostagger -Wl,-dy

multi_pos_cmdline: multi_pos_cmdline.cpp
	${cc} ${ccflags} -o multi_pos_cmdline multi_pos_cmdline.cpp \
		../thirdparty/tinythreadpp/tinythread.cpp \
		-I./ \
		-I../include/ \
		-I../thirdparty/tinythreadpp/ \
		-Wl,-dn -L../lib/ -lpostagger -Wl,-dy -lpthread

ner: ner.cpp
	${cc} ${ccflags} -o ner ner.cpp -I./ \
		-I../include \
		-L../lib/ -lner

par: par.cpp
	${cc} ${ccflags} -o par par.cpp -I./ \
		-I../include \
		-L../lib -lparser

par_cmdline: par_cmdline.cpp
	${cc} ${ccflags} -o par_cmdline par_cmdline.cpp -I./ \
		-I../include/ \
		-Wl,-dn -L../lib/ -lparser -Wl,-dy

srl: srl.cpp
	${cc} ${ccflags} -o srl srl.cpp -I./ \
		-I../include \
		-L../lib -lsrl

.PHONY: clean

clean:
	rm cws
	rm cws_cmdline
	rm multi_cws_cmdline
	rm pos
	rm pos_cmdline
	rm multi_pos_cmdline
	rm ner
	rm par
	rm par_cmdline
	rm srl
