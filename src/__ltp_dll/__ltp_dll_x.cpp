#include "__ltp_dll.h"
#pragma comment(lib, "__ltp_dll.lib")

namespace HIT_IR_LTP {
	// Get words
	LTP_DLL_API int _GetWordsFromSentence(const char **vecWord, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetWordsFromSentence(const char **vecWord, const int sz, int globalSentIdx);

	// Get POSs
	LTP_DLL_API int _GetPOSsFromSentence(const char **vecPOS, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetPOSsFromSentence(const char **vecPOS, const int sz, int globalSentIdx);

	// Get NEs
	LTP_DLL_API int _GetNEsFromSentence(const char **vecNE, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetNEsFromSentence(const char **vecNE, const int sz, int globalSentIdx);

	// Get WSDs
	/*
	LTP_DLL_API int _GetWSDsFromSentence(const char **vecWSD, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetWSDsFromSentence(const char **vecWSD, const int sz, int globalSentIdx);

	LTP_DLL_API int _GetWSDExplainsFromSentence(const char **vecWSDExplain, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetWSDExplainsFromSentence(const char **vecWSDExplain, const int sz, int globalSentIdx);
	*/

	// Get Parses
	LTP_DLL_API int _GetParsesFromSentence(pair<int, const char *> *vecParse, const int sz, int paragraphIdx, int sentenceIdx);
	LTP_DLL_API int _GetParsesFromSentence(pair<int, const char *> *vecParse, const int sz, int globalSentIdx);

	// Get SRL
	LTP_DLL_API int _GetPredArgToWord(	const char **vecType, pair<int, int> *vecBegEnd, const int sz,
		int paragraphIdx, int sentenceIdx, int wordIdx);
	LTP_DLL_API int _GetPredArgToWord(	const char **vecType, pair<int, int> *vecBegEnd, const int sz,
		int globalSentIdx, int wordIdx);
	LTP_DLL_API int _GetPredArgToWord(	const char **vecType, pair<int, int> *vecBegEnd, const int sz,
		int globalWordIdx);
}

namespace HIT_IR_LTP {
	// Get words
	int GetWordsFromSentence(vector<const char *> &vecWord, int paragraphIdx, int sentenceIdx)
	{
		vecWord.clear();

		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecWord.resize(wordNum);
			_GetWordsFromSentence(&vecWord[0], wordNum, paragraphIdx, sentenceIdx);
		}

		return 0;
	}

	int GetWordsFromSentence(vector<const char *> &vecWord, int globalSentIdx)
	{
		vecWord.clear();

		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecWord.resize(wordNum);
			_GetWordsFromSentence(&vecWord[0], wordNum, globalSentIdx);
		}

		return 0;
	}

	// Get POSs
	int GetPOSsFromSentence(vector<const char *> &vecPOS, int paragraphIdx, int sentenceIdx)
	{
		vecPOS.clear();
		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecPOS.resize(wordNum);
			_GetPOSsFromSentence(&vecPOS[0], wordNum, paragraphIdx, sentenceIdx);
		}

		return 0;
	}

	int GetPOSsFromSentence(vector<const char *> &vecPOS, int globalSentIdx)
	{
		vecPOS.clear();
		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecPOS.resize(wordNum);
			_GetPOSsFromSentence(&vecPOS[0], wordNum, globalSentIdx);
		}

		return 0;
	}

	// Get NEs
	int GetNEsFromSentence(vector<const char *> &vecNE, int paragraphIdx, int sentenceIdx)
	{
		vecNE.clear();
		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecNE.resize(wordNum);
			_GetNEsFromSentence(&vecNE[0], wordNum, paragraphIdx, sentenceIdx);
		}

		return 0;
	}

	int GetNEsFromSentence(vector<const char *> &vecNE, int globalSentIdx)
	{
		vecNE.clear();
		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecNE.resize(wordNum);
			_GetNEsFromSentence(&vecNE[0], wordNum, globalSentIdx);
		}

		return 0;
	}

	// Get WSDs
	/*
	int GetWSDsFromSentence(vector<const char *> &vecWSD, int paragraphIdx, int sentenceIdx)
	{
		vecWSD.clear();
		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecWSD.resize(wordNum);
			_GetWSDsFromSentence(&vecWSD[0], wordNum, paragraphIdx, sentenceIdx);
		}

		return 0;
	}

	int GetWSDsFromSentence(vector<const char *> &vecWSD, int globalSentIdx)
	{
		vecWSD.clear();
		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecWSD.resize(wordNum);
			_GetWSDsFromSentence(&vecWSD[0], wordNum, globalSentIdx);
		}

		return 0;
	}

	int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int paragraphIdx, int sentenceIdx)
	{
		vecWSDExplain.clear();
		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecWSDExplain.resize(wordNum);
			_GetWSDExplainsFromSentence(&vecWSDExplain[0], wordNum, paragraphIdx, sentenceIdx);
		}

		return 0;
	}

	int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int globalSentIdx)
	{
		vecWSDExplain.clear();
		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecWSDExplain.resize(wordNum);
			_GetWSDExplainsFromSentence(&vecWSDExplain[0], wordNum, globalSentIdx);
		}
		return 0;
	}
	*/

	// Get Parses
	int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int paragraphIdx, int sentenceIdx)
	{
		vecParse.clear();
		int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
		if (wordNum > 0)
		{
			vecParse.resize(wordNum);
			_GetParsesFromSentence(&vecParse[0], wordNum, paragraphIdx, sentenceIdx);
		}
		return 0;
	}

	int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int globalSentIdx)
	{
		vecParse.clear();
		int wordNum = CountWordInSentence(globalSentIdx);
		if (wordNum > 0)
		{
			vecParse.resize(wordNum);
			_GetParsesFromSentence(&vecParse[0], wordNum, globalSentIdx);
		}
		return 0;
	}

	// Get SRL
	int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd, 
							int paragraphIdx, int sentenceIdx, int wordIdx)
	{
		vecType.clear();
		vecBegEnd.clear();
		int argNum = CountPredArgToWord(paragraphIdx, sentenceIdx, wordIdx);
		if (argNum > 0)
		{
			vecType.resize(argNum);
			vecBegEnd.resize(argNum);
			_GetPredArgToWord(&vecType[0], &vecBegEnd[0], argNum, paragraphIdx, sentenceIdx, wordIdx);
		}
		return 0;
	}

	int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
							int globalSentIdx, int wordIdx)
	{
		vecType.clear();
		vecBegEnd.clear();
		int argNum = CountPredArgToWord(globalSentIdx, wordIdx);
		if (argNum > 0)
		{
			vecType.resize(argNum);
			vecBegEnd.resize(argNum);
			_GetPredArgToWord(&vecType[0], &vecBegEnd[0], argNum, globalSentIdx, wordIdx);
		}
		return 0;
	}

	int GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
							int globalWordIdx)
	{
		vecType.clear();
		vecBegEnd.clear();
		int argNum = CountPredArgToWord(globalWordIdx);
		if (argNum > 0)
		{
			vecType.resize(argNum);
			vecBegEnd.resize(argNum);
			_GetPredArgToWord(&vecType[0], &vecBegEnd[0], argNum, globalWordIdx);
		}
		return 0;
	}

}

