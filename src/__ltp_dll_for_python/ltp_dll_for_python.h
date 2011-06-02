#ifndef _LTP_DLL_FOR_PYTHON_H
#define _LTP_DLL_FOR_PYTHON_H

#ifdef LTP_DLL_FOR_PYTHON_EXPORT
#define LTP_DLL_FOR_PYTHON_API extern "C" _declspec(dllexport)
#else
#define LTP_DLL_FOR_PYTHON_API extern "C" _declspec(dllimport)
#endif

#include "../__ltp_dll/__ltp_dll.h"
#pragma comment(lib, "__ltp_dll.lib")

LTP_DLL_FOR_PYTHON_API int py_main2(const char *inFile, const char *outFile, const char* confFile = "ltp_modules_to_do.conf");

// DOM operation
LTP_DLL_FOR_PYTHON_API int CreateDOMFromTxt(const char *cszTxtFileName);
LTP_DLL_FOR_PYTHON_API int CreateDOMFromXml(const char *cszXmlFileName);
LTP_DLL_FOR_PYTHON_API int CreateDOMFromString(const char *str);

LTP_DLL_FOR_PYTHON_API int ClearDOM();
LTP_DLL_FOR_PYTHON_API int SaveDOM(const char *cszSaveFileName);

// Modules
LTP_DLL_FOR_PYTHON_API int SplitSentence();
//LTP_DLL_FOR_PYTHON_API int IRLAS();			// Word segment and POS
//LTP_DLL_FOR_PYTHON_API int SegmentWord();	// Word segment
LTP_DLL_FOR_PYTHON_API int CRFWordSeg();	// CRF-based word segment
LTP_DLL_FOR_PYTHON_API int PosTag();		// POSTagging
LTP_DLL_FOR_PYTHON_API int NER();			// Named entity recognition
//LTP_DLL_FOR_PYTHON_API int WSD();			// Word sense disambiguation
LTP_DLL_FOR_PYTHON_API int GParser();		// Dependency parser (Graph-based Method)
//LTP_DLL_FOR_PYTHON_API int Parser();		// Dependency parser (Ma Jinshan)
LTP_DLL_FOR_PYTHON_API int SRL();			// Semantic role labeling

// Counting
LTP_DLL_FOR_PYTHON_API int CountParagraphInDocument();

LTP_DLL_FOR_PYTHON_API int CountSentenceInParagraph(int paragraphIdx);
LTP_DLL_FOR_PYTHON_API int CountSentenceInDocument();

LTP_DLL_FOR_PYTHON_API int CountWordInSentence_p(int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int CountWordInSentence(int globalSentIdx);
LTP_DLL_FOR_PYTHON_API int CountWordInDocument();


// Get paragraph, NOTE: can ONLY used before split sentence.
LTP_DLL_FOR_PYTHON_API const char *GetParagraph(int paragraphIdx);

// Get sentence
LTP_DLL_FOR_PYTHON_API const char *GetSentence_p(int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API const char *GetSentence(int globalSentIdx);

// Get Word
LTP_DLL_FOR_PYTHON_API const char *GetWord_p_s(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetWord_s(int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetWord(int globalWordIdx);

// Get POS
LTP_DLL_FOR_PYTHON_API const char *GetPOS_p_s(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetPOS_s(int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetPOS(int globalWordIdx);

// Get NE
LTP_DLL_FOR_PYTHON_API const char *GetNE_p_s(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetNE_s(int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API const char *GetNE(int globalWordIdx);

// Get WSD
/*
LTP_DLL_FOR_PYTHON_API int	GetWSD_p_s(const char **p_wsd, const char **p_explain, int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int	GetWSD_s(const char **p_wsd, const char **p_explain, int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int	GetWSD(const char **p_wsd, const char **p_explain, int globalWordIdx);
*/

// Get Parser
LTP_DLL_FOR_PYTHON_API int	GetParse_p_s(int *p_parent, const char **p_relate, int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int	GetParse_s(int *p_parent, const char **p_relate, int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int	GetParse(int *p_parent, const char **p_relate, int globalWordIdx);

// Get words
LTP_DLL_FOR_PYTHON_API int GetWordsFromSentence_p(const char *word_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetWordsFromSentence(const char *word_arr[], int arr_size, int globalSentIdx);

// Get POSs
LTP_DLL_FOR_PYTHON_API int GetPOSsFromSentence_p(const char *pos_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetPOSsFromSentence(const char *pos_arr[], int arr_size, int globalSentIdx);

// Get NEs
LTP_DLL_FOR_PYTHON_API int GetNEsFromSentence_p(const char *ne_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetNEsFromSentence(const char *ne_arr[], int arr_size, int globalSentIdx);

// Get WSDs
/*
LTP_DLL_FOR_PYTHON_API int GetWSDsFromSentence_p(const char *wsd_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetWSDsFromSentence(const char *wsd_arr[], int arr_size, int globalSentIdx);

LTP_DLL_FOR_PYTHON_API int GetWSDExplainsFromSentence_p(const char *explain_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetWSDExplainsFromSentence(const char *explain_arr[], int arr_size, int globalSentIdx);
*/

// Get Parses
LTP_DLL_FOR_PYTHON_API int GetParsesFromSentence_p(int parent_arr[], const char *relate_arr[], int arr_size, int paragraphIdx, int sentenceIdx);
LTP_DLL_FOR_PYTHON_API int GetParsesFromSentence(int parent_arr[], const char *relate_arr[], int arr_size, int globalSentIdx);

// Get SRL
LTP_DLL_FOR_PYTHON_API int CountPredArgToWord_p_s(	int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int CountPredArgToWord_p(	int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int CountPredArgToWord(		int globalWordIdx);

LTP_DLL_FOR_PYTHON_API int GetPredArgToWord_p_s(const char *type_arr[], int beg_arr[], int end_arr[], int arr_size, 
												int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int GetPredArgToWord_p(	const char *type_arr[], int beg_arr[], int end_arr[], int arr_size,
												int globalSentIdx, int wordIdx);
LTP_DLL_FOR_PYTHON_API int GetPredArgToWord(	const char *type_arr[], int beg_arr[], int end_arr[], int arr_size,
												int globalWordIdx);

#endif