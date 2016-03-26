#ifndef __LTP_NER_DLL_H__
#define __LTP_NER_DLL_H__

#include <iostream>
#include <vector>

#define NER_DLL_API
#define NER_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef NER_DLL_API
#ifdef NER_DLL_API_EXPORT
    #define NER_DLL_API extern "C" _declspec(dllexport)
#else
    #define NER_DLL_API extern "C" _deslspec(dllimport)
    #pragma comment(lib, "ner.lib")
#endif      //  end for NER_DLL_API_EXPORT
#endif      //  end for _WIN32

/*
 * create a new parser
 *
 *  @param[in]  path    the path of the model
 *  @return     void *  the pointer to the segmentor
 */
NER_DLL_API void * ner_create_recognizer(const char * path);

/*
 * release the segmentor resources
 *
 *  @param[in]  segmentor   the segmentor
 *  @return     int         i don't know
 */
NER_DLL_API int ner_release_recognizer(void * ner);

/*
 * run segment on the given segmentor
 *
 *  @param[in]  line        the string to be segmented
 *  @param[out] words       the words of the input line
 *  @return     int         the number of word tokens, if the input arguments
 *                          is not legal, return 0
 */
NER_DLL_API int ner_recognize(void * ner,
                              const std::vector<std::string> & words,
                              const std::vector<std::string> & postags,
                              std::vector<std::string> & tags);

#endif  //  end for __LTP_NER_DLL_H__
