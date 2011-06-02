#define CRFWS_DLL_API_EXPORT
#include "CRFWS_DLL.h"
#include "CRFWS.h"
#include <iostream>
#include <string.h>
using namespace std;

CRFWS crfws;

int CRFWS_LoadResource(const char *path)
{
	int ret = 0;
	try {
		ret = crfws.CreateEngine(path);
	} catch (const exception &e) {
		ret = -2;
		cerr << e.what() << endl;
	}

	return ret;
}

int CRFWS_WordSegment_dll(const char* str, char** pWord, int& wordNum)
{
	wordNum = 0;

	string line = str;
	vector<string> vctWords;
	int ret = crfws.WordSegment(line, vctWords);
	wordNum = vctWords.size();
	for (int i = 0; i < wordNum; i++)
	{
		strcpy(pWord[i], vctWords[i].c_str());
		pWord[i][vctWords[i].size()] = '\0';
	}

	return 0;
}

void CRFWS_ReleaseResource()
{
	crfws.DeleteEngine();
}

