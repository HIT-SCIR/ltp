#ifndef __SRL_DLL_H__
#define __SRL_DLL_H__

//#define STL_USING_ALL
//#include <STL.h>
#include <string>
#include <vector>
#include <stdlib.h>

using namespace std;

#define SRL_DLL_API
#define SRL_DLL_API_EXPORT

#if defined(_MSC_VER)
    #undef SRL_DLL_API
    #ifdef SRL_DLL_API_EXPORT
        #define SRL_DLL_API extern "C" _declspec(dllexport)
    #else
        #define SRL_DLL_API extern "C" _declspec(dllimport)
        #pragma comment(lib, "srl.lib")
    #endif
#endif

SRL_DLL_API int SRL(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult
);

// Load Resources
SRL_DLL_API int SRL_LoadResource(const string &ConfigDir);

// Release Resources
SRL_DLL_API int SRL_ReleaseResource();

// Perform SRL
SRL_DLL_API int DoSRL(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult
);

SRL_DLL_API int GetSRLResult_size(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult);


SRL_DLL_API int GetSRLResult(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult);

#endif


