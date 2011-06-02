/*
HIT-IRLab (c) 2001-2005, all rights reserved.
This software is "XML Text Representation for NLP"
Its aim is to integrate all the modules of IRLab into a uniform frame
The author of this software if Huipeng Zhang (zhp@ir.hit.edu.cn)
The create time of this software is 2005-11-01
In this software, a open source XML parser TinyXML is used
We Thank to the author of it -- Lee Thomason
*/

#ifndef _XML4NLP_H_
#define _XML4NLP_H_

#pragma warning(disable : 4786 4267 4018)
//#include <cstring>
//#include <cassert>
//#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../__util/MyLib.h"
using namespace std;

#include "tinyxml.h"

// note types used in note operations
extern const char * const NOTE_SENT;
extern const char * const NOTE_WORD;
extern const char * const NOTE_POS;
extern const char * const NOTE_NE;
extern const char * const NOTE_PARSER;
extern const char * const NOTE_WSD;
extern const char * const NOTE_SRL;
extern const char * const NOTE_CLASS;
extern const char * const NOTE_SUM;
extern const char * const NOTE_CR;


/////////////////////////////////////////////////////////////////////////////////////
/// the class XML4NLP is derived from TiXmlDocument.
/////////////////////////////////////////////////////////////////////////////////////
class XML4NLP
{
public:
	XML4NLP();	
	virtual ~XML4NLP();

	// read a raw text and create a initial DOM tree 
	int CreateDOMFromFile(const char* fileName);
	int CreateDOMFromString(const string& str);
	int CreateDOMFromString(const char* str)
	{
		return CreateDOMFromString( string(str) );
	}

	// load a XML file and parse it into a DOM tree
	int LoadXMLFromFile(const char* fileName);
	int LoadXMLFromString(const string& str)
	{
		return LoadXMLFromString(str.c_str());
	}
	int LoadXMLFromString(const char* str);

	// clear and save the DOM tree
	void ClearDOM();
	int SaveDOM(const char* fileName);
	void SaveDOM(string &strDocument);
	
	// note operation
	bool QueryNote(const char *cszNoteName)  const;
	int SetNote(const char *cszNoteName);
	int ClearNote(const char *cszNoteName);
	void ClearAllNote();	

	// some counting functions
	int CountParagraphInDocument() const { return m_document_t.vecParagraph_t.size(); }
	
	int CountSentenceInParagraph(int paragraphIdx) const
	{
		if ( 0 != CheckRange(paragraphIdx) ) return 0;
		return m_document_t.vecParagraph_t[paragraphIdx].vecSentence_t.size();
	}
	int CountSentenceInDocument() const;
	
	int CountWordInSentence(int paragraphIdx, int sentenceIdx) const;
	int CountWordInSentence(int globalSentIdx) const;
	int CountWordInParagraph(int paragraphIdx) const;
	int CountWordInDocument() const;

	// get paragraph, sentence and word contents
	int			GetParagraph(int paragraphIdx, string &strParagraph) const;
	const char *GetParagraph(int paragraphIdx) const; // Only used when have not SplitSentence(), or will return NULL.
	
	const char *GetSentence(int paragraphIdx, int sentenceIdx) const;
	const char *GetSentence(int globalSentIdx) const;
	
	const char *GetWord(int paragraphIdx, int sentenceIdx, int wordIdx) const;
	const char *GetWord(int globalSentIdx, int wordIdx) const;
	const char *GetWord(int glabalWordIdx) const;

	const char *GetPOS(int paragraphIdx, int sentenceIdx, int wordIdx) const;
	const char *GetPOS(int globalSentIdx, int wordIdx) const;
	const char *GetPOS(int glabalWordIdx) const;

	const char *GetNE(int paragraphIdx, int sentenceIdx, int wordIdx) const;
	const char *GetNE(int globalSentIdx, int wordIdx) const;
	const char *GetNE(int glabalWordIdx) const;

	/*
	int	GetWSD(pair<const char *, const char *> &WSD_explain, int paragraphIdx, int sentenceIdx, int wordIdx) const;
	int	GetWSD(pair<const char *, const char *> &WSD_explain, int globalSentIdx, int wordIdx) const;
	int	GetWSD(pair<const char *, const char *> &WSD_explain, int glabalWordIdx) const;
	*/

	int	GetParse(pair<int, const char *> &parent_Relate, int paragraphIdx, int sentenceIdx, int wordIdx) const;
	int	GetParse(pair<int, const char *> &parent_Relate, int globalSentIdx, int wordIdx) const;
	int	GetParse(pair<int, const char *> &parent_Relate, int glabalWordIdx) const;

	// for sentence splitting
	int GetSentencesFromParagraph(vector<const char *> &vecSentence, int paragraphIdx) const;
	int GetSentencesFromParagraph(vector<string> &vecSents, int paragraphIdx) const;
	int SetSentencesToParagraph(const vector<string> &vecSents, int paragraphIdx);
	
	// for word segmentation
	int GetWordsFromSentence(vector<const char *> &vecWord, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWord, paragraphIdx, sentenceIdx, TAG_CONT);
	}
	int GetWordsFromSentence(vector<const char *> &vecWord, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecWord, globalSentIdx, TAG_CONT);
	}
	int GetWordsFromSentence(vector<string> &vecWord, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWord, paragraphIdx, sentenceIdx, TAG_CONT);
	}
	int GetWordsFromSentence(vector<string> &vecWord, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecWord, globalSentIdx, TAG_CONT);
	}
	int SetWordsToSentence(const vector<string> &vecWord, int paragraphIdx, int sentenceIdx);
	int SetWordsToSentence(const vector<string> &vecWord, int sentenceIdx);

	// for POS tagging
	int GetPOSsFromSentence(vector<const char *> &vecPOS, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
	}
	int GetPOSsFromSentence(vector<const char *> &vecPOS, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecPOS, globalSentIdx, TAG_POS);
	}
	int GetPOSsFromSentence(vector<string> &vecPOS, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
	}
	int GetPOSsFromSentence(vector<string> &vecPOS, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecPOS, globalSentIdx, TAG_POS);
	}
	int SetPOSsToSentence(const vector<string> &vecPOS, int paragraphIdx, int sentenceIdx)
	{
		return SetInfoToSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
	}
	int SetPOSsToSentence(const vector<string> &vecPOS, int sentenceIdx)
	{
		return SetInfoToSentence(vecPOS, sentenceIdx, TAG_POS);
	}
	
	// for NE
	int GetNEsFromSentence(vector<const char *> &vecNE, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
	}
	int GetNEsFromSentence(vector<const char *> &vecNE, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecNE, globalSentIdx, TAG_NE);
	}
	int GetNEsFromSentence(vector<string> &vecNE, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
	}
	int GetNEsFromSentence(vector<string> &vecNE, int globalSentIdx) const
	{
		return GetInfoFromSentence(vecNE, globalSentIdx, TAG_NE);
	}
	int SetNEsToSentence(const vector<string> &vecNE, int paragraphIdx, int sentenceIdx)
	{
		return SetInfoToSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
	}
	int SetNEsToSentence(const vector<string> &vecNE, int sentenceIdx)
	{
		return SetInfoToSentence(vecNE, sentenceIdx, TAG_NE);
	}
	
	// for WSD
	/*
	int GetWSDsFromSentence(vector<const char *> &vecWSD, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSD, paragraphIdx, sentenceIdx, TAG_WSD);
	}
	int GetWSDsFromSentence(vector<const char *> &vecWSD, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSD, sentenceIdx, TAG_WSD);
	}
	int GetWSDsFromSentence(vector<string> &vecWSD, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSD, paragraphIdx, sentenceIdx, TAG_WSD);
	}
	int GetWSDsFromSentence(vector<string> &vecWSD, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSD, sentenceIdx, TAG_WSD);
	}
	int SetWSDsToSentence(const vector<string> &vecWSD, int paragraphIdx, int sentenceIdx)
	{
		return SetInfoToSentence(vecWSD, paragraphIdx, sentenceIdx, TAG_WSD);
	}
	int SetWSDsToSentence(const vector<string> &vecWSD, int sentenceIdx)
	{
		return SetInfoToSentence(vecWSD, sentenceIdx, TAG_WSD);
	}

	int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSDExplain, paragraphIdx, sentenceIdx, TAG_WSD_EXP);
	}
	int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSDExplain, sentenceIdx, TAG_WSD_EXP);
	}
	int GetWSDExplainsFromSentence(vector<string> &vecWSDExplain, int paragraphIdx, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSDExplain, paragraphIdx, sentenceIdx, TAG_WSD_EXP);
	}
	int GetWSDExplainsFromSentence(vector<string> &vecWSDExplain, int sentenceIdx) const
	{
		return GetInfoFromSentence(vecWSDExplain, sentenceIdx, TAG_WSD_EXP);
	}
	int SetWSDExplainsToSentence(const vector<string> &vecWSDExplain, int paragraphIdx, int sentenceIdx)
	{
		return SetInfoToSentence(vecWSDExplain, paragraphIdx, sentenceIdx, TAG_WSD_EXP);
	}
	int SetWSDExplainsToSentence(const vector<string> &vecWSDExplain, int sentenceIdx)
	{
		return SetInfoToSentence(vecWSDExplain, sentenceIdx, TAG_WSD_EXP);
	}
	*/

	// for Parser
	int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int paragraphIdx, int sentenceIdx) const;
	int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, int sentenceIdx) const;
	int GetParsesFromSentence(vector< pair<int, string> > &vecParse, int paragraphIdx, int sentenceIdx) const;
	int GetParsesFromSentence(vector< pair<int, string> > &vecParse, int sentenceIdx) const;
	int SetParsesToSentence(const vector< pair<int, string> > &vecParse, int paragraphIdx, int sentenceIdx);
	int SetParsesToSentence(const vector< pair<int, string> > &vecParse, int sentenceIdx);
	int SetParsesToSentence(const vector<int> &vecHead, const vector<string> &vecRel, int paragraphIdx, int sentenceIdx);
	int SetParsesToSentence(const vector<int> &vecHead, const vector<string> &vecRel, int sentenceIdx);

	// for text summarization
	const char* GetTextSummary() const;
	int SetTextSummary(const char* textSum);

	// for text classification
	const char* GetTextClass() const;
	int SetTextClass(const char* textClass);

	// for SRL
	int CountPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx) const;
	int CountPredArgToWord(	int globalSentIdx, int wordIdx) const;
	int CountPredArgToWord(	int globalWordIdx) const;
	int GetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx,
							vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const;
	int GetPredArgToWord(	int globalSentIdx, int wordIdx,
							vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const;
	int GetPredArgToWord(	int globalWordIdx,
							vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const;
	int GetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx, 
							vector<string> &vecType, vector< pair<int, int> > &vecBegEnd) const;
	int GetPredArgToWord(	int sentenceIdx, int wordIdx, 
							vector<string> &vecType, vector< pair<int, int> > &vecBegEnd) const;
	int SetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx, 
							const vector<string> &vecType, const vector< pair<int, int> > &vecBegEnd);
	int SetPredArgToWord(	int sentenceIdx, int wordIdx,
							const vector<string> &vecType, const vector< pair<int, int> > &vecBegEnd);

	// for coreference resolution
	int CountEntity() const
	{
		return m_coref.vecEntity.size();
	}
	int CountMentionInEntity(int entityIdx)
	{
		if (entityIdx >= m_coref.vecEntity.size()) {
			cerr << "entity idx is too large" << endl;
			return -1;
		}
		return m_coref.vecEntity[entityIdx].vecMention.size();
	}
	int GetMentionOfEntity(vector< pair<int, int> > &vecMention, int entityIdx) const;
	int GetCoreference(vector< vector< pair<int, int> > >& vecCoref) const;
	int SetCoreference(const vector< vector< pair<int, int> > >& vecCoref);

public:
	int MapGlobalSentIdx2paraIdx_sentIdx(int sentenceIdx, pair<int, int> &paraIdx_sentIdx) const;
	int MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(int globalWordIdx, int &paraIdx, int &sentIdx, int &wordIdx) const;
	int CheckRange(int paragraphIdx, int sentenceIdx, int wordIdx) const;
	int CheckRange(int paragraphIdx, int sentenceIdx) const;
	int CheckRange(int paragraphIdx) const;

	void ReportTiXmlDocErr() const;

	int BuildParagraph(string &strParagraph, int paragraphIdx);

private:
	typedef struct {
		TiXmlElement *wordPtr;
	} Word_t;

	typedef struct {
		vector<Word_t> vecWord_t;
		TiXmlElement *sentencePtr;
	} Sentence_t;

	typedef struct {
		vector<Sentence_t> vecSentence_t;
		TiXmlElement *paragraphPtr;
	} Paragraph_t;

	typedef struct {
		vector<Paragraph_t> vecParagraph_t;
		TiXmlElement *documentPtr;
	} Document_t;

	typedef struct {
		TiXmlElement *nodePtr;
	} Note, Summary, TextClass;

	typedef struct {
		TiXmlElement *mentionPtr;
	} Mention;

	typedef struct {
		vector<Mention> vecMention;
		TiXmlElement *entityPtr;
	} Entity;

	typedef struct {
		vector<Entity> vecEntity;
		TiXmlElement *nodePtr;
	} Coref;

private:
	// initialization during loading txt
	int BuildDOMFrame();

	// initialization during loading xml
	int InitXmlStructure();
	void CheckNoteForOldLtml();
	int InitXmlDocument(Document_t &document);
	int InitXmlParagraph(vector<Paragraph_t> &vecParagraph, TiXmlElement *paragraphPtr);
	int InitXmlSentence(vector<Sentence_t> &vecSentence, TiXmlElement *stnsPtr);
	int InitXmlWord(vector<Word_t> &vecWord, TiXmlElement *wordPtr);
	int InitXmlCoref(Coref &coref);
	int InitXmlEntity(vector<Entity> &vecEntity, TiXmlElement *entityPtr);
	int InitXmlMention(vector<Mention> &vecMention, TiXmlElement *mentionPtr);

	int GetInfoFromSentence(vector<const char *> &vecInfo, int paragraphIdx, int sentenceIdx, const char *attrName) const;
	int GetInfoFromSentence(vector<const char *> &vecInfo, int sentenceIdx, const char *attrName) const;
	int GetInfoFromSentence(vector<string> &vec, int paragraphIdx, int sentenceIdx, const char* attrName) const;
	int GetInfoFromSentence(vector<string> &vec, int sentenceIdx, const char* attrName) const;
	int SetInfoToSentence(const vector<string> &vec, int paragraphIdx, int sentenceIdx, const char* attrName);
	int SetInfoToSentence(const vector<string> &vec, int sentenceIdx, const char* attrName);
	int SetInfoToSentence(const vector<int> &vec, int paragraphIdx, int sentenceIdx, const char* attrName);
	int SetInfoToSentence(const vector<int> &vec, int sentenceIdx, const char* attrName);

/*-------------------------------------------*/

private:
	vector<int> m_vecBegStnsIdxOfPara;
	vector<int> m_vecBegWordIdxOfStns;

	Document_t m_document_t;
	Note m_note;
	Summary m_summary;
	TextClass m_textclass;
	Coref m_coref;

	TiXmlDocument m_tiXmlDoc;

/*-------------------------------------------*/

private:
	static const char * const TAG_DOC;
	static const char * const TAG_NOTE;
	static const char * const TAG_SUM;
	static const char * const TAG_CLASS;
	static const char * const TAG_COREF;
	static const char * const TAG_COREF_MENT;
	static const char * const TAG_COREF_CR;
	static const char * const TAG_PARA;
	static const char * const TAG_SENT;
	static const char * const TAG_WORD;
	static const char * const TAG_CONT;		//sent, word
	static const char * const TAG_POS;
	static const char * const TAG_NE;
	static const char * const TAG_WSD;
	static const char * const TAG_WSD_EXP;
	static const char * const TAG_PSR_PARENT;
	static const char * const TAG_PSR_RELATE;
	static const char * const TAG_SRL_ARG;
	static const char * const TAG_SRL_TYPE;
	static const char * const TAG_BEGIN;	// cr, srl
	static const char * const TAG_END;		// cr, srl
	static const char * const TAG_ID;		// para, sent, word
};

#endif
