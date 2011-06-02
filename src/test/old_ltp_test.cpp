// Defines the entry point for the console application.
//

#include <stdlib.h>
#include <iostream>

#include "../__xml4nlp/Xml4nlp.h"
#include "../__ltp_dll/Ltp.h"

using namespace std;

static XML4NLP xml4nlp;
static LTP ltp(xml4nlp);

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		cerr << "Usage: ./ltp_test <type> <test_file>" << endl;
		exit(1);
	}

	cout << "Begin ..." << endl;
	string sentence;
	string type(argv[1]);
	ifstream in(argv[2]);

	if (!in.is_open())
	{
		cerr << "Cann't open file!" << endl;
		exit(1);
	}

	while(in >> sentence){
		cout << "Input sentence is: " << sentence << endl;

		xml4nlp.CreateDOMFromString(sentence);
		if(type == "ws"){
			ltp.crfWordSeg();
		} else if(type == "pos"){
			ltp.postag();
		} else if(type == "ner"){
			ltp.ner();
		} else if(type == "dp"){
			ltp.gparser();
		} else if(type == "srl"){
			ltp.srl();
		} else {
			ltp.srl();
		}

		string result;
		xml4nlp.SaveDOM(result);

		cout << "Result is: " << result << endl;
		xml4nlp.ClearDOM();
	}

	return 0;
}

