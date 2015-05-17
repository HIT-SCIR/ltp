// Defines the entry point for the console application.
//

#include <stdlib.h>
#include <iostream>

#include "../__xml4nlp/Xml4nlp.h"
#include "../__ltp_dll/Ltp.h"

using namespace std;

XML4NLP xml4nlp;
static LTP ltp;

int main(int argc, char *argv[])
{
	if (argc != 4)
	{
		cerr << "Usage: ./ltp_test <type> <test_xml_file> <result_file>" << endl;
		exit(1);
	}

	string type(argv[1]);

	xml4nlp.LoadXMLFromFile(argv[2]);
	if(type == "ws"){
		ltp.crfWordSeg(xml4nlp);
	} else if(type == "pos"){
		ltp.postag(xml4nlp);
	} else if(type == "ner"){
		ltp.ner(xml4nlp);
	} else if(type == "dp"){
		ltp.gparser(xml4nlp);
	} else if(type == "srl"){
		ltp.srl(xml4nlp);
	} else {
		ltp.srl(xml4nlp);
	}

	xml4nlp.SaveDOM(argv[3]);

	xml4nlp.ClearDOM();

	return 0;
}

