#ifndef __NER_DLL_H__
#define __NER_DLL_H__

#define NER_DLL_API
#define NER_DLL_API_EXPORT

#ifdef _WIN32 	
		#undef NER_DLL_API
		#ifdef NER_DLL_API_EXPORT
			#define NER_DLL_API extern "C" _declspec(dllexport) 
		#else
			#define NER_DLL_API extern "C" _declspec(dllimport)
			#pragma comment(lib, "ner.lib")
		#endif
#endif

#include <vector>
#include <string>
using std::vector; 
using std::string;

int NER(void* NETagger, const vector<string>& vecWord, const vector<string>& vecPOS, vector<string>& vecResult);


//load Resource
NER_DLL_API int NER_LoadResource(char* path);

//create an object for Convert Code
NER_DLL_API void* NER_CreateNErecoger();

//release Converter
NER_DLL_API void NER_ReleaseNErecoger(void* pNer);

NER_DLL_API void NER_ReleaseResource();

NER_DLL_API void NER_SetOption(int isEntity, int isTime, int isNum);

NER_DLL_API void NERtesting(void* pNer, char* pstrIn, char* pstrOut, int tagform);


#endif
