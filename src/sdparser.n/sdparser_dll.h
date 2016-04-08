#ifndef __LTP_SDPARSER_DLL_H__
#define __LTP_SDPARSER_DLL_H__

#include <iostream>
#include <vector>

#define SDPARSER_DLL_API
#define SDPARSER_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef SDPARSER_DLL_API
#ifdef SDPARSER_DLL_API_EXPORT
    #define SDPARSER_DLL_API extern "C" _declspec(dllexport)
#else
    #define SDPARSER_DLL_API extern "C" _deslspec(dllimport)
    #pragma comment(lib, "sdparser.lib")
#endif // end for SDPARSER_DLL_API
#endif // end for _WIN32

/*
 * create a new postagger
 *
 *  @param[in] path the path of the model
 *  @return void * the pointer to the segmentor
 */
SDPARSER_DLL_API void * sdparser_create_parser(const char * path);

/*
 * release the postagger resources
 *
 *  @param[in]  segmentor   the segmentor
 *  @return     int         i don't know
 */
SDPARSER_DLL_API int sdparser_release_parser(void * parser);

/*
 * run postag given the postagger on the input words
 *
 *  @param[in]  words       the string to be segmented
 *  @param[out] tags        the words of the input line
 *  @return     int         the number of word tokens, if input arguments
 *                          are not legal, return 0
 */
SDPARSER_DLL_API int sdparser_parse(void * parser,
        const std::vector< std::string > & words,
        const std::vector< std::string > & postags,
        std::vector<int> & heads,
        std::vector<std::string> & deprels);

#endif  //  end for __LTP_SDPARSER_DLL_H__
