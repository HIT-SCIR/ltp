#include "CRFWS_DLL.h"


int CRFWS_WordSegment_x(const string &sent, vector<string> &vecWord)
{
	int len = sent.size()+10;
	char** pWord = new char*[len];
	for (int j = 0; j < len; j++)
	{
		pWord[j] = new char[len];
	}

	int wordNum;
	int ret = CRFWS_WordSegment_dll( (char*)sent.c_str(), pWord, wordNum);

	for (int j = 0; j < wordNum; j++)
	{
		vecWord.push_back(pWord[j]);
	}

	for (int j = 0; j < len; j++)
	{
		delete[] pWord[j];
	}
	delete[] pWord;
	
	return ret;
}