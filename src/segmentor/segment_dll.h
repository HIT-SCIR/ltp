#ifndef __LTP_SEGMENTOR_DLL_H__
#define __LTP_SEGMENTOR_DLL_H__

#include <iostream>
#include <vector>

#define SEGMENTOR_DLL_API
#define SEGMENTOR_DLL_API_EXPORT

#ifdef _WIN32
#undef SEGMENTOR_DLL_API 
#ifdef SEGMENTOR_DLL_API_EXPORT
    #define SEGMENROR_DLL_API extern "C" _declspec(dllimport)
#else
    #define SEGMENTOR_DLL_API extern "C" _deslspec(dllimport)
    #pragma comment(lib, "segmentor.lib")
#endif      //  end for PARSER_DLL_API
#endif      //  end for _WIN32

/*
 * create a new parser
 *
 *  @param[in]  path    the path of the model
 *  @return     void *  the pointer to the segmentor
 */
SEGMENTOR_DLL_API void *   segmentor_create_segmentor(const char * path);

/*
 * release the segmentor resources
 *
 *  @param[in]  segmentor   the segmentor
 *  @return     int         i don't know
 */
SEGMENTOR_DLL_API int      segmentor_release_segmentor(void * parser); 

/*
 * run segment on the given segmentor
 *
 *  @param[in]  line        the string to be segmented
 *  @param[out] words       the words of the input line
 *  @return     int         the number of word tokens
 */
SEGMENTOR_DLL_API int      segmentor_segment(void * parser,
        const std::string & line,
        std::vector<std::string> & words);

#endif  //  end for __LTP_SEGMENTOR_DLL_H__
