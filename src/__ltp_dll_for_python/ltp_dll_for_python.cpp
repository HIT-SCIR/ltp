#define LTP_DLL_FOR_PYTHON_EXPORT

#include "ltp_dll_for_python.h"

#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

int py_main2(const char *inFile, const char *outFile, const char* confFile)
{
	return main2(inFile, outFile, confFile);
}

int CreateDOMFromTxt(const char *cszTxtFileName)
{
	return HIT_IR_LTP::CreateDOMFromTxt(cszTxtFileName);
}

int CreateDOMFromXml(const char *cszXmlFileName)
{
	return HIT_IR_LTP::CreateDOMFromXml(cszXmlFileName);
}

int CreateDOMFromString(const char *str)
{
	return HIT_IR_LTP::CreateDOMFromString(str);
}

int ClearDOM()
{
	return HIT_IR_LTP::ClearDOM();
}

int SaveDOM(const char *cszSaveFileName)
{
	return HIT_IR_LTP::SaveDOM(cszSaveFileName);
}

// Modules
int SplitSentence()
{
	return HIT_IR_LTP::SplitSentence();
}
/*
int IRLAS()			// Word segment and POS
{
	return HIT_IR_LTP::IRLAS();
}
*/
/*
int SegmentWord()	// Word segment
{
	return HIT_IR_LTP::SegmentWord();
}
*/
int CRFWordSeg()	// CRF-based Word segment
{
	return HIT_IR_LTP::CRFWordSeg();
}
int PosTag()		// POSTagging
{
	return HIT_IR_LTP::PosTag();
}
int NER()			// Named entity recognition
{
	return HIT_IR_LTP::NER();
}
/*
int WSD()			// Word sense disambiguation
{
	return HIT_IR_LTP::WSD();
}
*/
int GParser()		// Dependency parser
{
	return HIT_IR_LTP::GParser();
}
/*
int Parser()		// Dependency parser
{
	return HIT_IR_LTP::Parser();
}
*/
int SRL()			// Semantic role labeling
{
	return HIT_IR_LTP::SRL();
}

// Counting
int CountParagraphInDocument()
{
	return HIT_IR_LTP::CountParagraphInDocument();
}

int CountSentenceInParagraph(int paragraphIdx)
{
	return HIT_IR_LTP::CountSentenceInParagraph(paragraphIdx);
}
int CountSentenceInDocument()
{
	return HIT_IR_LTP::CountSentenceInDocument();
}

int CountWordInSentence_p(int paragraphIdx, int sentenceIdx)
{
	return HIT_IR_LTP::CountWordInSentence(paragraphIdx, sentenceIdx);
}
int CountWordInSentence(int globalSentIdx)
{
	return HIT_IR_LTP::CountWordInSentence(globalSentIdx);
}
int CountWordInDocument()
{
	return HIT_IR_LTP::CountWordInDocument();
}

// Get paragraph, NOTE: can ONLY used before split sentence.
const char *GetParagraph(int paragraphIdx)
{
	return HIT_IR_LTP::GetParagraph(paragraphIdx);
}

// Get sentence
const char *GetSentence_p(int paragraphIdx, int sentenceIdx)
{
	return HIT_IR_LTP::GetSentence(paragraphIdx, sentenceIdx);
}
const char *GetSentence(int globalSentIdx)
{
	return HIT_IR_LTP::GetSentence(globalSentIdx);
}

// Get Word
const char *GetWord_p_s(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return HIT_IR_LTP::GetWord(paragraphIdx, sentenceIdx, wordIdx);
}
const char *GetWord_s(int globalSentIdx, int wordIdx)
{
	return HIT_IR_LTP::GetWord(globalSentIdx, wordIdx);
}
const char *GetWord(int globalWordIdx)
{
	return HIT_IR_LTP::GetWord(globalWordIdx);
}

// Get POS
const char *GetPOS_p_s(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return HIT_IR_LTP::GetPOS(paragraphIdx, sentenceIdx, wordIdx);
}
const char *GetPOS_s(int globalSentIdx, int wordIdx)
{
	return HIT_IR_LTP::GetPOS(globalSentIdx, wordIdx);
}
const char *GetPOS(int globalWordIdx)
{
	return HIT_IR_LTP::GetPOS(globalWordIdx);
}

// Get NE
const char *GetNE_p_s(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return HIT_IR_LTP::GetNE(paragraphIdx, sentenceIdx, wordIdx);
}
const char *GetNE_s(int globalSentIdx, int wordIdx)
{
	return HIT_IR_LTP::GetNE(globalSentIdx, wordIdx);
}
const char *GetNE(int globalWordIdx)
{
	return HIT_IR_LTP::GetNE(globalWordIdx);
}

// Get WSD
/*
int	GetWSD_p_s(const char **p_wsd, const char **p_explain, int paragraphIdx, int sentenceIdx, int wordIdx)
{
	pair<const char *, const char *> wsd_explain;
	if (0 == HIT_IR_LTP::GetWSD(wsd_explain, paragraphIdx, sentenceIdx, wordIdx))
	{
		*p_wsd = wsd_explain.first;
		*p_explain = wsd_explain.second;
		return 0;
	}
	else
		return -1;
}
int	GetWSD_s(const char **p_wsd, const char **p_explain, int globalSentIdx, int wordIdx)
{
	pair<const char *, const char *> wsd_explain;
	if (0 == HIT_IR_LTP::GetWSD(wsd_explain, globalSentIdx, wordIdx))
	{
		*p_wsd = wsd_explain.first;
		*p_explain = wsd_explain.second;
		return 0;
	}
	else
		return -1;
}
int	GetWSD(const char **p_wsd, const char **p_explain, int globalWordIdx)
{
	pair<const char *, const char *> wsd_explain;
	if (0 == HIT_IR_LTP::GetWSD(wsd_explain, globalWordIdx))
	{
		*p_wsd = wsd_explain.first;
		*p_explain = wsd_explain.second;
		return 0;
	}
	else
		return -1;
}
*/

// Get Parser
int	GetParse_p_s(int *p_parent, const char **p_relate, int paragraphIdx, int sentenceIdx, int wordIdx)
{
	pair<int, const char *> parent_relate;
	if (0 == HIT_IR_LTP::GetParse(parent_relate, paragraphIdx, sentenceIdx, wordIdx))
	{
		*p_parent = parent_relate.first;
		*p_relate = parent_relate.second;
		return 0;
	}
	else
		return -1;
}
int	GetParse_s(int *p_parent, const char **p_relate, int globalSentIdx, int wordIdx)
{
	pair<int, const char *> parent_relate;
	if (0 == HIT_IR_LTP::GetParse(parent_relate, globalSentIdx, wordIdx))
	{
		*p_parent = parent_relate.first;
		*p_relate = parent_relate.second;
		return 0;
	}
	else
		return -1;
}
int	GetParse(int *p_parent, const char **p_relate, int globalWordIdx)
{
	pair<int, const char *> parent_relate;
	if (0 == HIT_IR_LTP::GetParse(parent_relate, globalWordIdx))
	{
		*p_parent = parent_relate.first;
		*p_relate = parent_relate.second;
		return 0;
	}
	else
		return -1;
}

// Get words
int GetWordsFromSentence_p(const char *word_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecWord;
	if (0 == HIT_IR_LTP::GetWordsFromSentence(vecWord, paragraphIdx, sentenceIdx))
	{
		if (vecWord.size() != arr_size)
		{
			cerr << "vecWord.size() != arr_size in GetWordsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecWord.begin(), vecWord.end(), word_arr);
		}
	}
	else
		return -1;

	return 0;
}

int GetWordsFromSentence(const char *word_arr[], int arr_size, int globalSentIdx)
{
	vector<const char *> vecWord;
	if (0 == HIT_IR_LTP::GetWordsFromSentence(vecWord, globalSentIdx))
	{
		if (vecWord.size() != arr_size)
		{
			cerr << "vecWord.size() != arr_size in GetWordsFromSentence()" << endl;
			return -1;
		}
		else
		{
			copy(vecWord.begin(), vecWord.end(), word_arr);
		}
	}
	else
		return -1;

	return 0;
}


// Get POSs
int GetPOSsFromSentence_p(const char *pos_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecPOS;
	if (0 == HIT_IR_LTP::GetPOSsFromSentence(vecPOS, paragraphIdx, sentenceIdx))
	{
		if (vecPOS.size() != arr_size)
		{
			cerr << "vecWord.size() != arr_size in GetPOSsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecPOS.begin(), vecPOS.end(), pos_arr);
		}
	}
	else
		return -1;

	return 0;
}
int GetPOSsFromSentence(const char *pos_arr[], int arr_size, int globalSentIdx)
{
	vector<const char *> vecPOS;
	if (0 == HIT_IR_LTP::GetPOSsFromSentence(vecPOS, globalSentIdx))
	{
		if (vecPOS.size() != arr_size)
		{
			cerr << "vecWord.size() != arr_size in GetPOSsFromSentence()" << endl;
			return -1;
		}
		else
		{
			copy(vecPOS.begin(), vecPOS.end(), pos_arr);
		}
	}
	else
		return -1;

	return 0;
}
// Get NEs
int GetNEsFromSentence_p(const char *ne_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecNE;
	if (0 == HIT_IR_LTP::GetNEsFromSentence(vecNE, paragraphIdx, sentenceIdx))
	{
		if (vecNE.size() != arr_size)
		{
			cerr << "vecNE.size() != arr_size in GetNEsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecNE.begin(), vecNE.end(), ne_arr);
		}
	}
	else
		return -1;

	return 0;
}
int GetNEsFromSentence(const char *ne_arr[], int arr_size, int globalSentIdx)
{
	vector<const char *> vecNE;
	if (0 == HIT_IR_LTP::GetNEsFromSentence(vecNE, globalSentIdx))
	{
		if (vecNE.size() != arr_size)
		{
			cerr << "vecNE.size() != arr_size in GetNEsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecNE.begin(), vecNE.end(), ne_arr);
		}
	}
	else
		return -1;

	return 0;
}

// Get WSDs
/*
int GetWSDsFromSentence_p(const char *wsd_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecWSD;
	if (0 == HIT_IR_LTP::GetWSDsFromSentence(vecWSD, paragraphIdx, sentenceIdx))
	{
		if (vecWSD.size() != arr_size)
		{
			cerr << "vecWSD.size() != arr_size in GetWSDsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecWSD.begin(), vecWSD.end(), wsd_arr);
		}
	}
	else
		return -1;

	return 0;
}
int GetWSDsFromSentence(const char *wsd_arr[], int arr_size, int globalSentIdx)
{
	vector<const char *> vecWSD;
	if (0 == HIT_IR_LTP::GetWSDsFromSentence(vecWSD, globalSentIdx))
	{
		if (vecWSD.size() != arr_size)
		{
			cerr << "vecWSD.size() != arr_size in GetWSDsFromSentence()" << endl;
			return -1;
		}
		else
		{
			copy(vecWSD.begin(), vecWSD.end(), wsd_arr);
		}
	}
	else
		return -1;

	return 0;
}

int GetWSDExplainsFromSentence_p(const char *explain_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecExplain;
	if (0 == HIT_IR_LTP::GetWSDExplainsFromSentence(vecExplain, paragraphIdx, sentenceIdx))
	{
		if (vecExplain.size() != arr_size)
		{
			cerr << "vecExplain.size() != arr_size in GetWSDExplainsFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			copy(vecExplain.begin(), vecExplain.end(), explain_arr);
		}
	}
	else
		return -1;

	return 0;
}
int GetWSDExplainsFromSentence(const char *explain_arr[], int arr_size, int globalSentIdx)
{
	vector<const char *> vecExplain;
	if (0 == HIT_IR_LTP::GetWSDExplainsFromSentence(vecExplain, globalSentIdx))
	{
		if (vecExplain.size() != arr_size)
		{
			cerr << "vecExplain.size() != arr_size in GetWSDExplainsFromSentence()" << endl;
			return -1;
		}
		else
		{
			copy(vecExplain.begin(), vecExplain.end(), explain_arr);
		}
	}
	else
		return -1;

	return 0;
}
*/

// Get Parses
int GetParsesFromSentence_p(int parent_arr[], const char *relate_arr[], int arr_size, int paragraphIdx, int sentenceIdx)
{
	vector< pair<int, const char *> > parent_relate;
	if (0 == HIT_IR_LTP::GetParsesFromSentence(parent_relate, paragraphIdx, sentenceIdx))
	{
		if (parent_relate.size() != arr_size)
		{
			cerr << "parent_relate.size() != arr_size in GetParsesFromSentence_p()" << endl;
			return -1;
		}
		else
		{
			int i = 0;
			for (; i<arr_size; ++i)
			{
				parent_arr[i] = parent_relate[i].first;
				relate_arr[i] = parent_relate[i].second;
			}
		}
	}
	else
		return -1;

	return 0;
}
int GetParsesFromSentence(int parent_arr[], const char *relate_arr[], int arr_size, int globalSentIdx)
{
	vector< pair<int, const char *> > parent_relate;
	if (0 == HIT_IR_LTP::GetParsesFromSentence(parent_relate, globalSentIdx))
	{
		if (parent_relate.size() != arr_size)
		{
			cerr << "parent_relate.size() != arr_size in GetParsesFromSentence()" << endl;
			return -1;
		}
		else
		{
			int i = 0;
			for (; i<arr_size; ++i)
			{
				parent_arr[i] = parent_relate[i].first;
				relate_arr[i] = parent_relate[i].second;
			}
		}
	}
	else
		return -1;

	return 0;
}

// Get SRL
int CountPredArgToWord_p_s(	int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return HIT_IR_LTP::CountPredArgToWord(paragraphIdx, sentenceIdx, wordIdx);
}
int CountPredArgToWord_s(	int globalSentIdx, int wordIdx)
{
	return HIT_IR_LTP::CountPredArgToWord(globalSentIdx, wordIdx);
}
int CountPredArgToWord(  int globalWordIdx)
{
	return HIT_IR_LTP::CountPredArgToWord(globalWordIdx);
}

int GetPredArgToWord_p_s(	const char *type_arr[], int beg_arr[], int end_arr[], int arr_size, 
							int paragraphIdx, int sentenceIdx, int wordIdx)
{
	vector<const char *> vecType;
	vector< pair<int, int> > vecBegEnd;
	if (0 == HIT_IR_LTP::GetPredArgToWord(vecType, vecBegEnd, paragraphIdx, sentenceIdx, wordIdx))
	{
		if (vecType.size() != arr_size)
		{
			cerr << "vecType.size() != arr_size in GetPredArgToWord_p_s()" << endl;
			return -1;
		}
		else
		{
			int i = 0;
			for (; i < arr_size; ++i)
			{
				type_arr[i] = vecType[i];
				beg_arr[i] = vecBegEnd[i].first;
				end_arr[i] = vecBegEnd[i].second;
			}
		}
	}
	else
		return -1;

	return 0;
}
int GetPredArgToWord_p(	const char *type_arr[], int beg_arr[], int end_arr[], int arr_size,
						int globalSentIdx, int wordIdx)
{
	vector<const char *> vecType;
	vector< pair<int, int> > vecBegEnd;
	if (0 == HIT_IR_LTP::GetPredArgToWord(vecType, vecBegEnd, globalSentIdx, wordIdx))
	{
		if (vecType.size() != arr_size)
		{
			cerr << "vecType.size() != arr_size in GetPredArgToWord_p()" << endl;
			return -1;
		}
		else
		{
			int i = 0;
			for (; i < arr_size; ++i)
			{
				type_arr[i] = vecType[i];
				beg_arr[i] = vecBegEnd[i].first;
				end_arr[i] = vecBegEnd[i].second;
			}
		}
	}
	else
		return -1;

	return 0;
}
int GetPredArgToWord(	const char *type_arr[], int beg_arr[], int end_arr[], int arr_size,
						int globalWordIdx)
{
	vector<const char *> vecType;
	vector< pair<int, int> > vecBegEnd;
	if (0 == HIT_IR_LTP::GetPredArgToWord(vecType, vecBegEnd, globalWordIdx))
	{
		if (vecType.size() != arr_size)
		{
			cerr << "vecType.size() != arr_size in GetPredArgToWord()" << endl;
			return -1;
		}
		else
		{
			int i = 0;
			for (; i < arr_size; ++i)
			{
				type_arr[i] = vecType[i];
				beg_arr[i] = vecBegEnd[i].first;
				end_arr[i] = vecBegEnd[i].second;
			}
		}
	}
	else
		return -1;

	return 0;
}

