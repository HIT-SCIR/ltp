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
        //#pragma comment(lib, "srl.lib")
    #endif
#endif
/**
 *
 * @param words
 * @param POSs
 * @param parse
 * @param vecSRLResult 谓词数组[<谓词序号, 论元数组[<论元label, <位置开始，位置结束>>]>] 序号从0开始
 * @return 执行情况 正常返回0 异常返回-1
 */
SRL_DLL_API int srl_dosrl(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<pair<int, string> > &parse,
        vector<pair<int, vector<pair<string, pair<int, int> > > > > &vecSRLResult
);

// Load Resources
SRL_DLL_API int srl_load_resource(const string &modelFile);

// Release Resources
SRL_DLL_API int srl_release_resource();

#endif


