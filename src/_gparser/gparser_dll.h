/////////////////////////////////////////////////////////////////////////////////////
// File Name   : gparser_dll.h
// Project Name: 
// Author      : Li Zhenghua
// Environment : Microsoft Visual .Net
// Description : Adapt interface to XML project
// Time        : 2008.12.13
// History     : 
// CopyRight   : HIT-IRLab (c) 2008-2012, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////


#ifndef __GPARSER_DLL_H__
#define __GPARSER_DLL_H__

#define GPARSER_DLL_API
#define GPARSER_DLL_API_EXPORT

#ifdef _WIN32
	#undef GPARSER_DLL_API
	#ifdef GPARSER_DLL_API_EXPORT
		#define GPARSER_DLL_API extern "C" _declspec(dllexport) 
	#else
		#define GPARSER_DLL_API extern "C" _declspec(dllimport)
		#pragma comment(lib, "_gparser.lib")
	#endif
#endif

#pragma warning(disable: 4786)

#include <string>
#include <vector>
using namespace std;

int GParser_Parse_x(void *gparser,
					  const vector<string> &vecWord,
					  const vector<string> &vecCPOS,
					  vector<int> &vecHead,
					  vector<string> &vecRel);

///////////////////////////////////////////////////////////////

GPARSER_DLL_API int GParser_Parse(void *gparser,
									  const vector<string>& vecWord,
									  const vector<string>& vecCPOS,
									  char *szHeads,
									  char *szLabels,
									  int &nHeadsSize,
									  int &nLablesSize);


GPARSER_DLL_API void *GParser_CreateParser(const char *szConfigFile);	// config file path+name
GPARSER_DLL_API int GParser_LoadResource(void *gparser, const char *szResourcePath);	// model file path
GPARSER_DLL_API int GParser_ReleaseParser(void *&gparser);
GPARSER_DLL_API int GParser_ReleaseResource(void *gparser);


#endif