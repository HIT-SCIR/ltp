#include "CRFWS.h"
#include "CRFSegEngine.h"


CRFWS::~CRFWS(void)
{
	if (engine) {
		delete engine;
	}
	engine = 0;
}

int CRFWS::CreateEngine(const char *path)
{
	cout << "Begin Load CRF model!" << endl;
	if (engine) delete engine;
	string strModel = path;
	strModel += "pku_all.model";
	engine = new CRFSegEngine( strModel.c_str() );
	if (!engine) {
		cerr << "crf-ws load model err: " << strModel << endl;	
		return -1;
	}
	cout << "Load CRF model over!" << endl;
	return 0;
}

int CRFWS::WordSegment(const string &line, vector<string> & vctWords)
{
	if (!engine) {
		cerr << "no crfws-engine created" << endl;
		return -1;
	}
	if (engine->Segment(line.c_str(), NULL, vctWords)) {
		return 0;
	} else {
		return -1;
	}
}

