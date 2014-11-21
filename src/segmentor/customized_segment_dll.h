/*
 * LTP Cutomized Segmentor Wrapper
 *
 *Allowing user to use segmentor like a C function
 */
#ifndef __LTP_CUSTOMIZED_SEGMENT_DLL_H__
#define __LTP_CUSTOMIZED_SEGMENT_DLL_H__

#include <iostream>
#include <vector>

#define CUSTOMIZED_SEGMENTOR_DLL_API
#define CUSTOMIZED_SEGMENTOR_DLL_API_EXPORT

#if defined(_MSC_VER)
#undef CUSTOMIZED_SEGMENTOR_DLL_API
#ifdef CUSTOMIZED_SEGMENTOR_DLL_API_EXPORT
  #define CUSTOMIZED_SEGMENT_DLL_API extern "C" _declspec(dllexport)
#else
  #define CUSTOMIZED_SEGMENTOR_DLL_API extern "C" _declspec(dllimport)
  #pragma comment(lib, "customized_segmentor.lib")
#endif    //  end for CUSTOMIZED_SEGMENTOR_DLL_API_EXPORT
#endif    //  end for _WIN32

CUSTOMIZED_SEGMENTOR_DLL_API int customized_segmentor_segment(const std::string & baseline_model_path,
                                                                 const std::string & model_path,
                                                                 const std::string & lexicon_path,
                                                                 const std::string & line,
                                                                 std::vector<std::string> & words);

#endif //  end for __LTP_CUSTOMIZED_SEGMENT_DLL_H__
