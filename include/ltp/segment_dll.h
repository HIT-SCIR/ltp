/*
 * LTP Segmentor Wrapper
 *
 * Allowing user to use segmentor like a C function.
 */
#ifndef __LTP_SEGMENT_DLL_H__
#define __LTP_SEGMENT_DLL_H__

#include <iostream>
#include <vector>

#define SEGMENTOR_DLL_API
#define SEGMENTOR_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef SEGMENTOR_DLL_API
#ifdef SEGMENTOR_DLL_API_EXPORT
  #define SEGMENTOR_DLL_API extern "C" _declspec(dllexport)
#else
  #define SEGMENTOR_DLL_API extern "C" _deslspec(dllimport)
  #pragma comment(lib, "segmentor.lib")
#endif    //  end for SEGMENTOR_DLL_API_EXPORT
#endif    //  end for _WIN32

/*
 * create a new parser
 *
 *  @param[in]  path          the path to the model
 *  @param[in]  lexicon_file  the path to the lexicon file
 *  @return     void *        the pointer to the segmentor
 */
SEGMENTOR_DLL_API void * segmentor_create_segmentor(const char * path,
                                                    const char * lexicon_file = NULL);

/*
 * release the segmentor resources
 *
 *  @param[in]  segmentor     the segmentor
 *  @return     int           i don't know
 */
SEGMENTOR_DLL_API int segmentor_release_segmentor(void * parser);

/*
 * run segment on the given segmentor
 *
 *  @param[in]  line        the string to be segmented
 *  @param[out] words       the words of the input line
 *  @return     int         the number of word tokens, if the input is not legal
 *                          return 0;
 */
SEGMENTOR_DLL_API int segmentor_segment(void * parser,
                                        const std::string & line,
                                        std::vector<std::string> & words);

SEGMENTOR_DLL_API void* customized_segmentor_create_segmentor(
    const char* path1,
    const char* path2,
    const char* lexicon_file = NULL);

SEGMENTOR_DLL_API int customized_segmentor_release_segmentor(void* segmentor);

SEGMENTOR_DLL_API int customized_segmentor_segment(
    void* segmentor,
    const std::string& line,
    std::vector<std::string>& words);

#endif  //  end for __LTP_SEGMENT_DLL_H__
