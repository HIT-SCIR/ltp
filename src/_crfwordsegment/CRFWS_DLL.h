#ifndef __CRFWS_DLL_H__
#define __CRFWS_DLL_H__

#include <string>
#include <vector>
#include <stdlib.h>


using namespace std;

#define CRFWS_DLL_API

#ifdef _WIN32
#undef CRFWS_DLL_API
#ifdef CRFWS_DLL_API_EXPORT
#define CRFWS_DLL_API extern "C" _declspec(dllexport)
#else
#define CRFWS_DLL_API extern "C" _declspec(dllimport)
#pragma comment(lib, "_crfwordsegment.lib")
#endif
#endif

//分词的dll接口
CRFWS_DLL_API int CRFWS_LoadResource(const char *path);
CRFWS_DLL_API int CRFWS_WordSegment_dll(const char* str, char **pWord, int &wordNum);
CRFWS_DLL_API void CRFWS_ReleaseResource();

int CRFWS_WordSegment_x(const string &sent, vector<string> &vecWord);

#endif


