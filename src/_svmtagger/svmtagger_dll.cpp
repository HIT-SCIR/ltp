#define SVMTAGGER_DLL_API_EXPORT

#include <iostream>
#include "svmtagger_dll.h"

#include "tagger.h"
#include "common.h"
#include "dict.h"
#include "er.h"
//#include "hash.h"
#include "list.h"
#include "stack.h"
//#include "swindow.h"
#include "weight.h"


tagger *pTagger = NULL;

int svmtagger_LoadResource(const char* szResPath)
{
	erCompRegExp();
	pTagger = new tagger("CHINES",szResPath);	
	pTagger->taggerPutStrategy(5);
	pTagger->taggerPutFlow("LR");
	pTagger->taggerInit(szResPath);

	return 0;
}
int svmtagger_PosTag(const char **szWordsArr,char **pword,int nWordNum)
{
	pTagger->taggerInitSw(const_cast<char **>(szWordsArr),pword,nWordNum);
	pTagger->taggerRun();
  
	return 0;
}
int svmtagger_ReleaseResource()
{
	delete pTagger;
	pTagger = 0;
	erFreeRegExp();
	return 0;
}
