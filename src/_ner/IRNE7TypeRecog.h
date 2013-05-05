#ifndef __IRNE7TYPERECOG_H__
#define __IRNE7TYPERECOG_H__

#include "crfpp.h"
#include <string>
#include <string.h>
#include <stdlib.h>
using std::string;

class IRNErecog
{
public:
	IRNErecog();
	~IRNErecog();
	void crf_set(char *model_path);
	void IRNE7TypeRecog(const string& strSen, string& StrOut, int tagForm, bool* isNEtypeFlag);

private:
	CRFPP::Tagger *tagger;
};


#endif
