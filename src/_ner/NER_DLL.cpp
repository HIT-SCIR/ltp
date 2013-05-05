#define NER_DLL_API_EXPORT
#include "NER_DLL.h"
#include "IRNE7TypeRecog.h"
#include <string.h>

/* old version:
   #include "Model.h"
   #include "InitDic.h"
*/

/* old version
   CModel pmodel;
*/

/* old version:
   InitDic dic;
*/

char *model_path;
int g_isEntity = 1;
int g_isTime = 1;
int g_isNum = 1;

bool* NEtypeFlag;

int NER_LoadResource(char* path)
{
    /* old version:
	string pathname = path;
	dic.loadRule(path);
	cout << "loadRule over" << endl;
	pmodel.LoadMEModel(pathname);
	cout << "LoadMEModel over" << endl; */

    model_path = path;
    return 1;
}

void* NER_CreateNErecoger()
{
	IRNErecog *pNER = new IRNErecog;
	/* old version: replace next line
	   pNER->setObject(&dic, &pmodel); */
	if (!pNER)
	    fprintf(stderr, "NER_CreateNErecoger: create failed.\n");
	pNER->crf_set(model_path);
	return pNER;	
}


void NER_ReleaseResource()
{
    /* old version:
	dic.releaseRes();
    */
    //delete pNer->tagger;
}

void NER_ReleaseNErecoger(void* pNer)
{
	delete pNer;
	/* old version:
	  delete[] NEtypeFlag;
	*/
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
	/* debug
	fprintf(stderr, "NERtesting: %s\n", pstrOut); */
}

void NER_SetOption(int isEntity, int isTime, int isNum)
{
	g_isEntity = isEntity;
	g_isTime = isTime;
	g_isNum = isNum;
}
