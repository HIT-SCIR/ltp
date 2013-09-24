all: cws cws_cmdline pos par ner multi_cws_cmdline

cws: cws.cpp
	g++ -o cws cws.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-L../lib/ -lsegmentor -lboost_regex

cws_cmdline: cws_cmdline.cpp
	g++ -o cws_cmdline cws_cmdline.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-Wl,-dn -L../lib/ -lsegmentor -lboost_regex -Wl,-dy

pos: pos.cpp
	g++ -o pos pos.cpp -I./ \
		-I../include/ \
		-L../lib/ -lpostagger

ner: ner.cpp
	g++ -o ner ner.cpp -I./ \
		-I../src/ner/ \
		-L../lib/ -lner

par: par.cpp
	g++ -o par par.cpp -I./ \
		-I../src/parser/ \
		-L../lib -lparser

multi_cws_cmdline: multi_cws_cmdline.cpp
	g++ -o multi_cws_cmdline multi_cws_cmdline.cpp \
		thirdparty/tinythreadpp/tinythread.cpp \
		-I./ \
		-I../include/ \
		-I../thirdparty/boost/include/ \
		-I./thirdparty/tinythreadpp/ \
		-Wl,-dn -L../lib/ -lsegmentor -lboost_regex -Wl,-dy -lpthread

.PHONY: clean

clean:
	rm cws
	rm cws_cmdline
	rm pos
	rm ner
	rm par
	rm multi_cws_cmdline
