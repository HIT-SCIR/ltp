all: cws pos par ner multi_cws

cws: cws.cpp
	g++ -o cws cws.cpp -I./ \
		-I../include/ \
		-I../thirdparty/boost/include \
		-L../lib/ -lsegmentor -lboost_regex

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

multi_cws: multi_cws.cpp
	g++ -o multi_cws multi_cws.cpp \
		thirdparty/tinythreadpp/tinythread.cpp \
		-I./ \
		-I../include/ \
		-I../thirdparty/boost/include/ \
		-I./thirdparty/tinythreadpp/ \
		-L../lib/ -lsegmentor -lboost_regex -lpthread

.PHONY: clean

clean:
	rm cws
	rm pos
	rm ner
	rm par
	rm multi_cws
