#pragma once

#include "SegEngine.h"
#include <string>

using namespace std;
using namespace las;

class CRFWS
{
public:
	CRFWS(void) : engine(0) {}
	~CRFWS(void);
	int CreateEngine(const char *path);
	int WordSegment(const string & line, vector<string> & vctWords);
	void DeleteEngine() {
		if (engine) {
			delete engine;
			engine = 0;
		}
	}

private:
	SegEngine	*engine;
};
