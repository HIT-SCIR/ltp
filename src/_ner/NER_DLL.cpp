#define NER_DLL_API_EXPORT
#include "NER_DLL.h"
#include "IRNE7TypeRecog.h"
#include <string.h>

char *model_path;
int g_isEntity = 1;
int g_isTime = 1;
int g_isNum = 1;

bool* NEtypeFlag;

int NER_LoadResource(char* path)
{
    model_path = path;
    return 1;
}

void* NER_CreateNErecoger()
{
	IRNErecog *pNER = new IRNErecog;
	if (!pNER)
	    fprintf(stderr, "NER_CreateNErecoger: create failed.\n");
	pNER->crf_set(model_path);
	return pNER;	
}


void NER_ReleaseResource()
{
}

void NER_ReleaseNErecoger(void* pNer)
{
	delete pNer;
}


void NERtesting(void* pNer, char* pstrIn,
				char* pstrOut, int tagform)
{
	IRNErecog* pner = (IRNErecog*)pNer;
	string strIn = pstrIn;
	string strOut;
	NEtypeFlag = new bool[3];

	NEtypeFlag[0] = g_isEntity == 1 ? true : false; 
	NEtypeFlag[1] = g_isTime == 1 ? true : false; 
	NEtypeFlag[2] = g_isNum == 1 ? true : false; 

	pner->IRNE7TypeRecog(strIn, strOut, tagform, NEtypeFlag);

	strcpy(pstrOut, strOut.c_str());
}

void NER_SetOption(int isEntity, int isTime, int isNum)
{
	g_isEntity = isEntity;
	g_isTime = isTime;
	g_isNum = isNum;
}
