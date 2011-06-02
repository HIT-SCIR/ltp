#define LTP_DLL_EXPORT
#include "__ltp_dll.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;
#include "Ltp.h"

//ofstream ltp_log_file("ltp_log.txt");

namespace HIT_IR_LTP {
	static XML4NLP g_xml4nlp;
	static LTP g_ltp(g_xml4nlp);
}

int main2(const char *inFile, const char *outFile, const char* confFile)
{
	ifstream config(confFile);
	if (!config.is_open())
	{
		cerr << "open ltp config file error: " << confFile << endl;
		return -100;
	}

	unsigned int flag = 0;
	string strLine;
	int i = 0;
	while ( getline(config, strLine) )
	{
		unsigned int tmp = atoi(strLine.substr(0, 1).c_str());
		flag |= (tmp << i);
		++i;
	}

	config.close();
	return HIT_IR_LTP::g_ltp.main2(inFile, outFile, flag);
}

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
/*
// Get words
LTP_DLL_API int _GetWordsFromSentence(vector<const char *> &vecWord, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetWordsFromSentence(vector<const char *> &vecWord, int globalSentIdx);

// Get POSs
LTP_DLL_API int _GetPOSsFromSentence(vector<const char *> &vecPOS, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetPOSsFromSentence(vector<const char *> &vecPOS, int globalSentIdx);

// Get NEs
LTP_DLL_API int _GetNEsFromSentence(vector<const char *> &vecNE, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetNEsFromSentence(vector<const char *> &vecNE, int globalSentIdx);

// Get WSDs
LTP_DLL_API int _GetWSDsFromSentence(vector<const char *> &vecWSD, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetWSDsFromSentence(vector<const char *> &vecWSD, int globalSentIdx);

LTP_DLL_API int _GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int globalSentIdx);

// Get Parses
LTP_DLL_API int _GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int paragraphIdx, int sentenceIdx);
LTP_DLL_API int _GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int globalSentIdx);

// Get SRL
LTP_DLL_API int _GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
									int paragraphIdx, int sentenceIdx, int wordIdx);
LTP_DLL_API int _GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
									int globalSentIdx, int wordIdx);
LTP_DLL_API int _GetPredArgToWord(	vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd,
									int globalWordIdx);
*/
}

// ------------------ IMPLIMENTATION --------------------

namespace HIT_IR_LTP {
int CreateDOMFromTxt(const char *cszTxtFileName)
{
	return g_ltp.CreateDOMFromTxt(cszTxtFileName);
}

int CreateDOMFromXml(const char *cszXmlFileName)
{
	return g_ltp.CreateDOMFromXml(cszXmlFileName);
}
/*
int CreateDOMFromString(const string &str)
{
	return g_xml4nlp.CreateDOMFromString(str);
}
*/
int CreateDOMFromString(const char *str)
{
	return g_xml4nlp.CreateDOMFromString(str);
}

int ClearDOM()
{
	g_xml4nlp.ClearDOM();
	return 0;
}

int SaveDOM(const char *cszSaveFileName)
{
	return g_ltp.SaveDOM(cszSaveFileName);
}

int SplitSentence()
{
	return g_ltp.splitSentence();
}

/*
int SegmentWord()
{
	return g_ltp.segmentWord();
}
*/

int CRFWordSeg()
{
	return g_ltp.crfWordSeg();
}

int PosTag()
{
	return g_ltp.postag();
}

int NER()
{
	return g_ltp.ner();
}

int GParser()
{
	return g_ltp.gparser();
}

int SRL()
{
	return g_ltp.srl();
}

// Counting
int CountParagraphInDocument()
{
	return g_xml4nlp.CountParagraphInDocument();
}

int CountSentenceInParagraph(int paragraphIdx)
{
	return g_xml4nlp.CountSentenceInParagraph(paragraphIdx);
}

int CountSentenceInDocument()
{
	return g_xml4nlp.CountSentenceInDocument();
}

int CountWordInSentence(int paragraphIdx, int sentenceIdx)
{
	return g_xml4nlp.CountWordInSentence(paragraphIdx, sentenceIdx);
}

int CountWordInSentence(int sentenceIdx)
{
	return g_xml4nlp.CountWordInSentence(sentenceIdx);
}

/*
int CountWordInParagraph(int paragraphIdx)
{
	return g_xml4nlp.CountWordInParagraph(paragraphIdx);
}
*/
int CountWordInDocument()
{
	return g_xml4nlp.CountWordInDocument();
}
/*
int CountWordInParagraph(int paragraphIdx)
{
	return g_xml4nlp.CountWordInParagraph(paragraphIdx);
}

int CountWordInDocument()
{
	return g_xml4nlp.CountWordInDocument();
}
*/

// Get paragraph
const char *GetParagraph(int paragraphIdx)
{
	return g_xml4nlp.GetParagraph(paragraphIdx);
}

// Get sentence
const char *GetSentence(int paragraphIdx, int sentenceIdx)
{
	return g_xml4nlp.GetSentence(paragraphIdx, sentenceIdx);
}

const char *GetSentence(int sentenceIdx)
{
	return g_xml4nlp.GetSentence(sentenceIdx);
}


// Get Word
const char *GetWord(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.GetWord(paragraphIdx, sentenceIdx, wordIdx);
}

const char *GetWord(int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.GetWord(sentenceIdx, wordIdx);
}

const char *GetWord(int wordIdx)
{
	return g_xml4nlp.GetWord(wordIdx);
}

// Get POS
const char *GetPOS(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.GetPOS(paragraphIdx, sentenceIdx, wordIdx);
}
const char *GetPOS(int globalSentIdx, int wordIdx)
{
	return g_xml4nlp.GetPOS(globalSentIdx, wordIdx);
}
const char *GetPOS(int globalWordIdx)
{
	return g_xml4nlp.GetPOS(globalWordIdx);
}

// Get NE
const char *GetNE(int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.GetNE(paragraphIdx, sentenceIdx, wordIdx);
}
const char *GetNE(int globalSentIdx, int wordIdx)
{
	return g_xml4nlp.GetNE(globalSentIdx, wordIdx);
}
const char *GetNE(int globalWordIdx)
{
	return g_xml4nlp.GetNE(globalWordIdx);
}

// Get Parser
int	GetParse(pair<int, const char *> &parent_relate, int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.GetParse(parent_relate, paragraphIdx, sentenceIdx, wordIdx);
}
int	GetParse(pair<int, const char *> &parent_relate, int globalSentIdx, int wordIdx)
{
	return g_xml4nlp.GetParse(parent_relate, globalSentIdx, wordIdx);
}
int	GetParse(pair<int, const char *> &parent_relate, int globalWordIdx)
{
	return g_xml4nlp.GetParse(parent_relate, globalWordIdx);
}

// Get words
int _GetWordsFromSentence(const char **arrWord, const int sz, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecWord(sz);
	if (0 != g_xml4nlp.GetWordsFromSentence(vecWord, paragraphIdx, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) {
		arrWord[i] = vecWord[i];
	}
	return 0;
}

int _GetWordsFromSentence(const char **arrWord, const int sz, int sentenceIdx)
{
	vector<const char *> vecWord(sz);
	if (0 != g_xml4nlp.GetWordsFromSentence(vecWord, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) {
		arrWord[i] = vecWord[i];
	}
	return 0;
}

/*
int _GetWordsFromSentence(vector<const char *> &vecWord, int paragraphIdx, int sentenceIdx)
{
	return g_xml4nlp.GetWordsFromSentence(vecWord, paragraphIdx, sentenceIdx);
}

int _GetWordsFromSentence(vector<const char *> &vecWord, int sentenceIdx)
{
	return g_xml4nlp.GetWordsFromSentence(vecWord, sentenceIdx);
}
*/

// Get POSs
int _GetPOSsFromSentence(const char **arrPOS, const int sz, int paragraphIdx, int sentenceIdx) {
	vector<const char *> vecPOS(sz);
	if (0 != g_xml4nlp.GetPOSsFromSentence(vecPOS, paragraphIdx, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrPOS[i] = vecPOS[i];
	return 0;
}

int _GetPOSsFromSentence(const char **arrPOS, const int sz, int sentenceIdx) {
	vector<const char *> vecPOS(sz);
	if (0 != g_xml4nlp.GetPOSsFromSentence(vecPOS, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrPOS[i] = vecPOS[i];
	return 0;
}

// Get NEs
int _GetNEsFromSentence(const char **arrNE, const int sz, int paragraphIdx, int sentenceIdx)
{
	vector<const char *> vecNE(sz);
	if (0 != g_xml4nlp.GetNEsFromSentence(vecNE, paragraphIdx, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrNE[i] = vecNE[i];
	return 0;
}

int _GetNEsFromSentence(const char **arrNE, const int sz, int sentenceIdx)
{
	vector<const char *> vecNE(sz);
	if (0 != g_xml4nlp.GetNEsFromSentence(vecNE, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrNE[i] = vecNE[i];
	return 0;
}

// Get Parses
int _GetParsesFromSentence(pair<int, const char *> *arrParse, const int sz, int paragraphIdx, int sentenceIdx)
{
	vector< pair<int, const char *> > vecParse(sz);
	if (0 != g_xml4nlp.GetParsesFromSentence(vecParse, paragraphIdx, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrParse[i] = vecParse[i];
	return 0;
}

int _GetParsesFromSentence(pair<int, const char *> *arrParse, const int sz, int sentenceIdx)
{
	vector< pair<int, const char *> > vecParse(sz);
	if (0 != g_xml4nlp.GetParsesFromSentence(vecParse, sentenceIdx)) return -1;
	for (int i = 0; i < sz; ++i) arrParse[i] = vecParse[i];
	return 0;
}

// Get SRL
int CountPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.CountPredArgToWord(paragraphIdx, sentenceIdx, wordIdx);
}
int CountPredArgToWord(	int sentenceIdx, int wordIdx)
{
	return g_xml4nlp.CountPredArgToWord(sentenceIdx, wordIdx);
}
int CountPredArgToWord( int globalWordIdx)
{
	return g_xml4nlp.CountPredArgToWord(globalWordIdx);
}

int _GetPredArgToWord(	const char **arrType, pair<int, int> *arrBegEnd, const int sz,
						int paragraphIdx, int sentenceIdx, int wordIdx)
{
	vector<const char *> vecType(sz);
	vector< pair<int, int> > vecBegEnd(sz);
	if (0 != g_xml4nlp.GetPredArgToWord(paragraphIdx, sentenceIdx, wordIdx, vecType, vecBegEnd)) return -1;
	for (int i = 0; i < sz; ++i) {
		arrType[i] = vecType[i];
		arrBegEnd[i] = vecBegEnd[i];
	}
	return 0;
}
int _GetPredArgToWord(	const char **arrType, pair<int, int> *arrBegEnd, const int sz,
					  int globalSentIdx, int wordIdx)
{
	vector<const char *> vecType(sz);
	vector< pair<int, int> > vecBegEnd(sz);
	if (0 != g_xml4nlp.GetPredArgToWord(globalSentIdx, wordIdx, vecType, vecBegEnd)) return -1;
	for (int i = 0; i < sz; ++i) {
		arrType[i] = vecType[i];
		arrBegEnd[i] = vecBegEnd[i];
	}
	return 0;
}
int _GetPredArgToWord(	const char **arrType, pair<int, int> *arrBegEnd, const int sz,
					  int globalWordIdx)
{
	vector<const char *> vecType(sz);
	vector< pair<int, int> > vecBegEnd(sz);
	if (0 != g_xml4nlp.GetPredArgToWord(globalWordIdx, vecType, vecBegEnd)) return -1;
	for (int i = 0; i < sz; ++i) {
		arrType[i] = vecType[i];
		arrBegEnd[i] = vecBegEnd[i];
	}
	return 0;
}

}
