#ifndef _LTP_DLL_H
#define _LTP_DLL_H

#ifdef LTP_DLL_EXPORT
#define LTP_DLL_API _declspec(dllexport)
#else
#define LTP_DLL_API _declspec(dllimport)
#endif

#pragma warning(disable : 4786)
#include <string>
#include <vector>
#include <utility>
using namespace std;

// Process the text file named as "inFile", and save the result into an xml file named as "outFile"
// "confFile" is used to tell the program to call which modules.
LTP_DLL_API int main2(const char *inFile, const char *outFile, const char* confFile = "ltp_modules_to_do.conf");	

namespace HIT_IR_LTP {

// DOM operation
LTP_DLL_API int CreateDOMFromTxt(const char *cszTxtFileName);
LTP_DLL_API int CreateDOMFromXml(const char *cszXmlFileName);
LTP_DLL_API int CreateDOMFromString(const char *str);
inline		int CreateDOMFromString(const string &str)	// Due to incompatible between VC6 and VC7.1 in DLL
				{ return CreateDOMFromString(str.c_str()); }
LTP_DLL_API int ClearDOM();
LTP_DLL_API int SaveDOM(const char *cszSaveFileName);

// Modules
LTP_DLL_API int SplitSentence();
//LTP_DLL_API int SegmentWord();	// Word segment
LTP_DLL_API int CRFWordSeg();	// CRF based Word segment
LTP_DLL_API int PosTag();		// Part-of-speech tagging
LTP_DLL_API int NER();			// Named entity recognition
LTP_DLL_API int GParser();		// Dependency parser (Graph-based Method)
LTP_DLL_API int SRL();			// Semantic role labeling

// Counting
LTP_DLL_API int CountParagraphInDocument();

LTP_DLL_API int CountSentenceInParagraph(int paragraphIdx);
LTP_DLL_API int CountSentenceInDocument();

LTP_DLL_API int CountWordInSentence(int paragraphIdx, int sentenceIdx);
LTP_DLL_API int CountWordInSentence(int globalSentIdx);
LTP_DLL_API int CountWordInDocument();


// Get paragraph, NOTE: can ONLY used before split sentence.
LTP_DLL_API const char *GetParagraph(int paragraphIdx);

// Get sentence
LTP_DLL_API const char *GetSentence(int paragraphIdx, int sentenceIdx);
LTP_DLL_API const char *GetSentence(int globalSentIdx);

// Get Word
LTP_DLL_API const char *GetWord(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API const char *GetWord(int globalSentIdx, int wordIdx);
LTP_DLL_API const char *GetWord(int globalWordIdx);

// Get POS
LTP_DLL_API const char *GetPOS(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API const char *GetPOS(int globalSentIdx, int wordIdx);
LTP_DLL_API const char *GetPOS(int globalWordIdx);

// Get NE
LTP_DLL_API const char *GetNE(int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API const char *GetNE(int globalSentIdx, int wordIdx);
LTP_DLL_API const char *GetNE(int globalWordIdx);

// Get Parser
LTP_DLL_API int	GetParse(pair<int, const char *> &parent_relate, int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API int	GetParse(pair<int, const char *> &parent_relate, int globalSentIdx, int wordIdx);
LTP_DLL_API int	GetParse(pair<int, const char *> &parent_relate, int globalWordIdx);

// Get words
int GetWordsFromSentence(vector<const char *> &vecWord, int paragraphIdx, int sentenceIdx);
int GetWordsFromSentence(vector<const char *> &vecWord, int globalSentIdx);

// Get POSs
int GetPOSsFromSentence(vector<const char *> &vecPOS, int paragraphIdx, int sentenceIdx);
int GetPOSsFromSentence(vector<const char *> &vecPOS, int globalSentIdx);

// Get NEs
int GetNEsFromSentence(vector<const char *> &vecNE, int paragraphIdx, int sentenceIdx);
int GetNEsFromSentence(vector<const char *> &vecNE, int globalSentIdx);

// Get Parses
int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int paragraphIdx, int sentenceIdx);
int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int globalSentIdx);

// Get SRL
LTP_DLL_API int CountPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API int CountPredArgToWord(	int globalSentIdx, int wordIdx);
LTP_DLL_API int CountPredArgToWord( int globalWordIdx);
int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd, 
						int paragraphIdx, int sentenceIdx, int wordIdx);
int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
						int globalSentIdx, int wordIdx);
int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
						int globalWordIdx);
}

#endif