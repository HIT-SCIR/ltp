// Defines the entry point for the console application.
//

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "Xml4nlp.h"
#include "Ltp.h"

using namespace std;

static XML4NLP xml4nlp;
static LTP ltp(xml4nlp);

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: ./ltp_test <type> <test_file> <result_file>" << endl;
        exit(1);
    }

    string type(argv[1]);
    string in_file(argv[2]);
    string res_file(argv[3]);

    xml4nlp.CreateDOMFromFile(in_file.c_str());

    if (type == "ws") {
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

    ofstream out(res_file.c_str());
    out << result << endl;
    cerr << "Results saved to " << res_file << endl;

    xml4nlp.ClearDOM();

    return 0;
}

