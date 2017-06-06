#ifndef __LTP_SPLIT_SENTENCE_H__
#define __LTP_SPLIT_SENTENCE_H__

#define SPLIT_SENTENCE_DLL_API
#define SPLIT_SENTENCE_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef SPLIT_SENTENCE_DLL_API
#ifdef SPLIT_SENTENCE_DLL_API_EXPORT
    #define SPLIT_SENTENCE_DLL_API extern "C" _declspec(dllexport)
#else
    #define SPLIT_SENTENCE_DLL_API extern "C" _deslspec(dllimport)
    //#pragma comment(lib, "splitsnt.lib")
#endif // end for SPLIT_SENTENCE_DLL_API
#endif // end for _WIN32

#include <string>
#include <vector>

// return (int)vecSentence.size();
/**
 * API interface for spliting sentences.
 *
 *  @param[in]  paragraph   The string to the paragraph
 *  @param[out] sentences   The splited sentences
 */
SPLIT_SENTENCE_DLL_API int SplitSentence(const std::string & paragraph,
    std::vector<std::string> & sentences);

#endif //__SPLIT_SENTENCE_H__

