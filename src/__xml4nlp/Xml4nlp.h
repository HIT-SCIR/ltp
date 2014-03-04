/*
 * HIT-IRLab (c) 2001-2005, all rights reserved.
 * This software is "XML Text Representation for NLP"
 * Its aim is to integrate all the modules of IRLab into a uniform frame
 * The author of this software if Huipeng Zhang (zhp@ir.hit.edu.cn)
 * The create time of this software is 2005-11-01
 * In this software, a open source XML parser TinyXML is used
 * We Thank to the author of it -- Lee Thomason
 */
#ifndef __LTP_XML4NLP_H__
#define __LTP_XML4NLP_H__

#pragma warning(disable : 4786 4267 4018)
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
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
class XML4NLP {
public:
  XML4NLP();
  virtual ~XML4NLP();

  // --------------------------------------------------------------
  // Functions for DOM Tree Creation
  // --------------------------------------------------------------
  /*
   * Create DOM from file, read in each line of the file and store
   * them in the xml tree.
   *
   *  @param[in]  filename  the filename
   */
  int CreateDOMFromFile(const char * filename);

  /*
   * Create DOM from raw string text.
   *
   *  @param[in]  str     the string
   *  @return     int     0 on success, otherwise -1
   */
  int CreateDOMFromString(const std::string & str);

  /*
   * A wrapper of CreateDOMFromString(const std::string & str);
   *
   *  @param[in]  str     the string
   *  @return     int     0 on success, otherwise -1
   */
  int CreateDOMFromString(const char * str);

  /*
   * Load XML DOM from file
   *
   *  @param[in]  filename  the file name
   *  @return     int     0 on success, otherwise -1
   */
  int LoadXMLFromFile(const char * fileName);

  /*
   * Load XML DOM from string
   *
   *  @param[in]  str     the string
   *  @return     int     0 on success, otherwise -1
   */
  int LoadXMLFromString(const char * str);

   /*
   * Load XML DOM from string
   *
   *  @param[in]  str     the string
   */
  int LoadXMLFromString(const std::string & str);

  /*
   * Clear the DOM tree
   */
  void ClearDOM();

  /*
   * Save the DOM tree to file
   *
   *  @param[in]  filename  the filename
   *  @return   int     0 on success, otherwise -1
   */
  int SaveDOM(const char * fileName);

  /*
   * Save the DOM tree to strin
   *
   *  @param[out] strDocument the str
   */
  void SaveDOM(string &strDocument) const;

  /*
   * Get attributes value in `<note/>`
   *
   *  @param[in]  note_name   the name of the attribute in note
   *  @return     bool        return true on `<note/>` exists and attributes
   *                          value equals "y", otherwise false.
   */
  bool QueryNote(const char * note_name) const;

  /*
   * Set attributes value in `<note/>` to "y"
   *
   *  @param[in]  note_name   the name of the attribute in note
   *  @return     int         return 0
   */
  int SetNote(const char * note_name);

  /*
   * Set attributes value in `<note/>` to "n"
   *
   *  @param[in]  cszNoteName the note name
   *  @return   int     return 0
   */
  int ClearNote(const char * note_name);

  /*
   * Set all nlp attributes value in `<note/>` to "n"
   *
   *  @param[in]  cszNoteName the note name
   *  @return   int     return 0
   */
  void ClearAllNote();

  // counting operation
  /*
   * count number of paragraph in document
   *
   *  @return   int   the number of paragraph
   */
  int CountParagraphInDocument() const;

  /*
   * conut number of sentence in paragraph
   *
   *  @param[in]  pid  the index number of paragraph
   *  @return   int       the number of paragraph
   */
  int CountSentenceInParagraph(int pid) const;

  /*
   * count number of all sentences in document
   *
   *  @return   int   the number of all sentences in document
   */
  int CountSentenceInDocument() const;

  /*
   * Count number of words in sentence, given the index of paragraph
   * and index of sentence.
   *
   *  @param[in]  pid   the index of paragraph
   *  @param[in]  sid   the index of sentence
   *  @return     int
   */
  int CountWordInSentence(int pid, int sid) const;

  /*
   * Count number of words in sentence, given the global index
   * of the sentence
   *
   *  @param[in]  global_sid    the global index of a sentence
   *  @return     int           number of sentence
   */
  int CountWordInSentence(int global_sid) const;

  /*
   * Count number of words in paragraph
   *
   *  @param[in]  pid     the index of paragraph
   *  @return     int     number of words in paragraph if legal
   *                      pid is given, otherwise -1
   */
  int CountWordInParagraph(int pid) const;

  /*
   * Count total number of words in paragraph
   *
   *  @return   int     number of words
   */
  int CountWordInDocument() const;

  /*
   * Get content of paragraph and store it in string
   *
   *  @param[in]  pid           the index of paragraph
   *  @param[out] strParagraph  the output string
   *  @return     int           0 on success, otherwise -1
   */
  int GetParagraph(int pid, string & strParagraph) const;

  /*
   * Get content of paragraph
   *
   *  @param[in]  pid           the index of paragraph
   *  @return     const char *  the pointer to the string, NULL on failure
   */
  const char * GetParagraph(int pid) const;

  /*
   * Get content of sentence
   *
   *  @param[in]  pid           the index of paragraph
   *  @param[in]  sid           the index of sentence
   *  @return     const char *  the pointer to the string, NULL on failure
   */
  const char * GetSentence(int pid, int sid) const;

  /*
   * Get content of sentence, given the sentence's global index
   *
   *  @param[in]  global_sid    the global index of the sentence
   *  @return   const char *  the pointer to the string, NULL on failure
   */
  const char * GetSentence(int global_sid) const;

  /*
   * Get word content
   *
   *  @param[in]  pid           the index of paragraph in document
   *  @param[in]  sid           the index of sentence in paragraph
   *  @param[in]  wid           the index of word in sentence
   *  @return     const char *  the pointer to the string, NULL on failure
   */
  const char * GetWord(int pid, int sid, int wid) const;

  /*
   * Get word content, given the global sentence index
   *
   *  @param[in]  global_sid    the global index of the sentence
   *  @param[in]  wid           the index of word in sentence
   *  @return     const char *  the pointer to the string, NULL on failure
   */
  const char * GetWord(int global_sid, int wid) const;

  /*
   * Get word content, given the global index of word
   *
   *  @param[in]  global_wid    the global index of the sentence
   *  @return     const char *  the pointer to the string, NULL on failure
   */
  const char * GetWord(int glabal_wid) const;

  /*
   * Get word's postag
   *
   *  @param[in]  pid           the index of the paragraph
   *  @param[in]  sid           the index of the sentence
   *  @param[in]  wid           the index of the word
   *  @return     const char *  the pointer to the string, NULL on failure.
   */
  const char * GetPOS(int pid, int sid, int wid) const;

  /*
   * Get word's postag
   *
   *  @param[in]  global_sid    the global index of sentence
   *  @param[in]  wid       the index of the word
   *  @return   const char *  the pointer to the string, NULL on failure.
   */
  const char * GetPOS(int global_sid, int wid) const;

  /*
   * Get word's postag, given the global index of the word in the document.
   *
   *  @param[in]  global_wid    the global index of the word.
   *  @return   const char *  the pointer to the string, NULL on failure.
   */
  const char * GetPOS(int global_wid) const;

  /*
   * Get word's NER tag
   *
   *  @param[in]  pid       the index of paragraph
   *  @param[in]  sid       the index of sentence
   *  @param[in]  wid       the index of word
   *  @return   const char *  the pointer to the tag, NULL on failure.
   */
  const char * GetNE(int pid, int sid, int wid) const;

  /*
   * Get word's NER tag, given the global index of sentence in the document.
   *
   *  @param[in]  global_sid    the global index of sentence
   *  @param[in]  wid       the index of the word
   *  @return   const char *  the pointer to the tag, NULL on failure.
   */
  const char * GetNE(int global_sid, int wid) const;

  /*
   * Get word's NER, given the global index of the word in the document.
   *
   *  @param[in]  global_wid    the global index of the word.
   *  @return   const char *  the pointer to the string, NULL on failure.
   */
  const char * GetNE(int glabalWordIdx) const;

  /*
   * Get word's WSD result (WSD module is under construction)
   *
   *  @param[out] WSD_explanation the explanation of the WSD
   *  @param[in]  pid       the index of paragraph
   *  @param[in]  sid       the index of sentence
   *  @param[in]  wid       the index of word
   *  @return   int       0 on success, otherwise -1
   */
  int GetWSD(pair<const char *, const char *> & WSD_explanation,
      int pid,
      int sid,
      int wid) const;

  /*
   * Get word's WSD result (WSD module is under construction)
   *
   *  @param[out] WSD_explanation the explanation of the WSD
   *  @param[in]  global_sid    the global index of sentence
   *  @param[in]  wid       the index of the word
   *  @return   int       0 on success, -1 on illegal index
   */
  int GetWSD(pair<const char *, const char *> & WSD_explanation,
      int global_sid,
      int wid) const;

  /*
   * Get word's WSD result (WSD module is under construction)
   *
   *  @param[out] WSD_explanation the explanation of the WSD
   *  @param[in]  global_wid      the global index of sentence
   *  @return     int             0 on success, -1 on illegal index
   */
  int GetWSD(pair<const char *, const char *> & WSD_explanation,
      int global_wid) const;

  /*
   * Get word's parsing result
   *
   *  @param[out] parent_relation the (parent, relation) pair
   *  @param[in]  pid       the index of paragraph
   *  @param[in]  sid       the index of sentence
   *  @param[in]  wid       the index of word
   *  @return   int       0 on success, -1 on illegal index
   */
  int GetParse(pair<int, const char *> & parent_relation,
      int pid,
      int sid,
      int wid) const;

  /*
   * Get word's parsing result
   *
   *  @param[out] parent_relation the (parent, relation) pair
   *  @param[in]  global_sid    the global index of sentence
   *  @param[in]  wid       the index of the word
   *  @return   int       0 on success, -1 on illegal index
   */
  int GetParse(pair<int, const char *> & parent_relation,
      int global_sid,
      int wid) const;

  /*
   * Get word's parsing result
   *
   *  @param[out] parent_relation the (parent, relation) pair
   *  @param[in]  global_wid    the global index of sentence
   *  @return   int       0 on success, -1 on illegal index
   */
  int GetParse(pair<int, const char *> &parent_relation,
      int glabal_wid) const;

  /*
   * Get sentences from paragraph
   *
   *  @param[out] vecSentence   the output vector
   *  @param[in]  paragraphIdx  the index to the paragraph
   */
  int GetSentencesFromParagraph(vector<const char *> & vecSentence,
      int paragraphIdx) const;

  /*
   * Get sentences from paragraph
   *
   *  @param[out] vectSentence  the output vector
   *  @param[in]  paragraphIdx  the index to the paragraph
   */
  int GetSentencesFromParagraph(vector<string> &vecSents,
      int paragraphIdx) const;

  int SetSentencesToParagraph(const vector<string> &vecSents,
      int paragraphIdx);

  /*
   * Get words from sentence
   *
   *  @param[out] vecWord     the word vector
   *  @param[in]  paragraphIdx  the index of paragraph
   *  @param[in]  sentenceIdx   the index of sentence
   */
  int GetWordsFromSentence(vector<const char *> &vecWord,
      int paragraphIdx,
      int sentenceIdx) const;

  /*
   * Get words from sentence
   *
   *  @param[out] vecWord     the word vector
   *  @param[in]  globalSentIdx   the global index of sentence
   */
  int GetWordsFromSentence(vector<const char *> &vecWord,
      int globalSentIdx) const;

  /*
   * Get words from sentence, std::string interface
   *
   *  @param[out] vecWord     the word vector
   *  @param[in]  paragraphIdx  the index of paragraph
   *  @param[in]  sentenceIdx   the index of sentence
   */
  int GetWordsFromSentence(vector<string> &vecWord,
      int paragraphIdx,
      int sentenceIdx) const;

  /*
   * Get words from sentence, std::string interface
   *
   *  @param[out] vecWord     the word vector
   *  @param[in]  globalSentIdx   the global index of sentence
   */
  int GetWordsFromSentence(vector<string> &vecWord,
      int globalSentIdx) const;

  /*
   * Set word to sentence
   *
   *  @param[in]  vecWord     the words
   *  @param[in]  paragraphIdx  the index of paragraph
   *  @param[in]  sentenceIdx   the index of sentence
   */
  int SetWordsToSentence(const vector<string> &vecWord,
      int paragraphIdx,
      int sentenceIdx);

  /*
   * Set word to sentence
   *
   *  @param[in]  vecWord     the words
   *  @param[in]  sentenceIdx   the global index of sentence
   */
  int SetWordsToSentence(const vector<string> &vecWord,
      int sentenceIdx);

  // for POS tagging
  int GetPOSsFromSentence(vector<const char *> & vecPOS,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetPOSsFromSentence(vector<const char *> & vecPOS,
      int globalSentIdx) const;

  int GetPOSsFromSentence(vector<string> & vecPOS,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetPOSsFromSentence(vector<string> & vecPOS,
      int globalSentIdx) const;

  int SetPOSsToSentence(const vector<string> & vecPOS,
      int paragraphIdx,
      int sentenceIdx);

  int SetPOSsToSentence(const vector<string> & vecPOS,
      int sentenceIdx);

  // for NE
  int GetNEsFromSentence(vector<const char *> &vecNE,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetNEsFromSentence(vector<const char *> &vecNE,
      int globalSentIdx) const;

  int GetNEsFromSentence(vector<string> &vecNE,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetNEsFromSentence(vector<string> &vecNE,
      int globalSentIdx) const;

  int SetNEsToSentence(const vector<string> &vecNE,
      int paragraphIdx,
      int sentenceIdx);

  int SetNEsToSentence(const vector<string> &vecNE,
      int sentenceIdx);

  int GetWSDsFromSentence(vector<const char *> &vecWSD,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetWSDsFromSentence(vector<const char *> &vecWSD,
      int sentenceIdx) const;

  int GetWSDsFromSentence(vector<string> &vecWSD,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetWSDsFromSentence(vector<string> &vecWSD,
      int sentenceIdx) const;

  int SetWSDsToSentence(const vector<string> &vecWSD,
      int paragraphIdx,
      int sentenceIdx);

  int SetWSDsToSentence(const vector<string> & vecWSD,
      int sentenceIdx);

  int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetWSDExplainsFromSentence(vector<const char *> &vecWSDExplain,
      int sentenceIdx) const;

  int GetWSDExplainsFromSentence(vector<string> &vecWSDExplain,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetWSDExplainsFromSentence(vector<string> &vecWSDExplain,
      int sentenceIdx) const;

  int SetWSDExplainsToSentence(const vector<string> &vecWSDExplain,
      int paragraphIdx,
      int sentenceIdx);

  int SetWSDExplainsToSentence(const vector<string> &vecWSDExplain,
      int sentenceIdx);

  // for Parser
  int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetParsesFromSentence(vector< pair<int, const char *> > &vecParse,
      int sentenceIdx) const;

  int GetParsesFromSentence(vector< pair<int, string> > &vecParse,
      int paragraphIdx,
      int sentenceIdx) const;

  int GetParsesFromSentence(vector< pair<int, string> > &vecParse,
      int sentenceIdx) const;

  int SetParsesToSentence(const vector< pair<int, string> > &vecParse,
      int paragraphIdx,
      int sentenceIdx);

  int SetParsesToSentence(const vector< pair<int, string> > &vecParse,
      int sentenceIdx);

  int SetParsesToSentence(const vector<int> &vecHead,
      const vector<string> &vecRel,
      int paragraphIdx,
      int sentenceIdx);

  int SetParsesToSentence(const vector<int> &vecHead,
      const vector<string> &vecRel,
      int sentenceIdx);

  // for text summarization
  const char* GetTextSummary() const;
  int SetTextSummary(const char* textSum);

  // for text classification
  const char* GetTextClass() const;
  int SetTextClass(const char* textClass);

  // for SRL
  int CountPredArgToWord(int paragraphIdx,
      int sentenceIdx,
      int wordIdx) const;

  int CountPredArgToWord(int globalSentIdx,
      int wordIdx) const;

  int CountPredArgToWord(int globalWordIdx) const;

  int GetPredArgToWord(int paragraphIdx,
      int sentenceIdx,
      int wordIdx,
      vector<const char *> &vecType,
      vector< pair<int, int> > &vecBegEnd) const;

  int GetPredArgToWord(int globalSentIdx,
      int wordIdx,
      vector<const char *> &vecType,
      vector< pair<int, int> > &vecBegEnd) const;

  int GetPredArgToWord(int globalWordIdx,
      vector<const char *> &vecType,
      vector< pair<int, int> > &vecBegEnd) const;

  int GetPredArgToWord(int paragraphIdx,
      int sentenceIdx,
      int wordIdx,
      vector<string> &vecType,
      vector< pair<int, int> > &vecBegEnd) const;

  int GetPredArgToWord(int sentenceIdx,
      int wordIdx,
      vector<string> &vecType,
      vector< pair<int, int> > &vecBegEnd) const;

  int SetPredArgToWord(int paragraphIdx,
      int sentenceIdx,
      int wordIdx,
      const vector<string> &vecType,
      const vector< pair<int, int> > &vecBegEnd);

  int SetPredArgToWord(int sentenceIdx,
      int wordIdx,
      const vector<string> &vecType,
      const vector< pair<int, int> > &vecBegEnd);

  // for coreference resolution
  int CountEntity() const;

  int CountMentionInEntity(int entityIdx);

  int GetMentionOfEntity(vector< pair<int, int> > &vecMention,
      int entityIdx) const;

  int GetCoreference(vector< vector< pair<int, int> > >& vecCoref) const;

  int SetCoreference(const vector< vector< pair<int, int> > >& vecCoref);

public:
  int DecodeGlobalId(int global_sid, int & pid, int & sid) const;

  int DecodeGlobalId(int globalWordIdx,
      int &paraIdx,
      int &sentIdx,
      int &wordIdx) const;

  int CheckRange(int paragraphIdx,
      int sentenceIdx,
      int wordIdx) const;

  int CheckRange(int paragraphIdx,
      int sentenceIdx) const;

  int CheckRange(int paragraphIdx) const;

  void ReportTiXmlDocErr() const;

  int BuildParagraph(string &strParagraph,
      int paragraphIdx);

private:
  typedef struct {
    TiXmlElement *wordPtr;
  } Word;

  typedef struct {
    vector<Word> words;
    TiXmlElement * sentencePtr;
  } Sentence;

  typedef struct {
    vector<Sentence>  sentences;
    TiXmlElement *    paragraphPtr;
  } Paragraph;

  typedef struct {
    vector<Paragraph> paragraphs;
    TiXmlElement *    documentPtr;
  } Document;

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

  typedef std::pair<const char *, const char *> WSDResult;
  typedef std::pair<int, const char *>          ParseResult;
private:
  // initialization during loading txt
  int BuildDOMFrame();

  // initialization during loading xml
  int InitXmlStructure();

  int InitXmlDocument(Document & document);

  int InitXmlParagraph(vector<Paragraph> & vecParagraph,
      TiXmlElement *paragraphPtr);

  int InitXmlSentence(vector<Sentence> &vecSentence,
      TiXmlElement *stnsPtr);

  int InitXmlWord(vector<Word> &vecWord,
      TiXmlElement *wordPtr);

  int InitXmlCoref(Coref &coref);

  int InitXmlEntity(vector<Entity> &vecEntity,
      TiXmlElement *entityPtr);

  int InitXmlMention(vector<Mention> &vecMention,
      TiXmlElement *mentionPtr);

  int GetInfoFromSentence(vector<const char *> &vecInfo,
      int paragraphIdx,
      int sentenceIdx,
      const char *attrName) const;

  int GetInfoFromSentence(vector<const char *> &vecInfo,
      int sentenceIdx,
      const char *attrName) const;

  int GetInfoFromSentence(vector<string> &vec,
      int paragraphIdx,
      int sentenceIdx,
      const char* attrName) const;

  int GetInfoFromSentence(vector<string> &vec,
      int sentenceIdx,
      const char * attrName) const;

  int SetInfoToSentence(const vector<string> &vec,
      int paragraphIdx,
      int sentenceIdx,
      const char * attrName);

  int SetInfoToSentence(const vector<string> &vec,
      int sentenceIdx,
      const char * attrName);

  int SetInfoToSentence(const vector<int> &vec,
      int paragraphIdx,
      int sentenceIdx,
      const char* attrName);

  int SetInfoToSentence(const vector<int> &vec,
      int sentenceIdx,
      const char* attrName);

  bool LTMLValidation();
  /*-------------------------------------------*/

private:
  vector<int> m_vecBegStnsIdxOfPara;
  vector<int> m_vecBegWordIdxOfStns;

  Document    document;
  Note        note;
  Summary     summary;
  TextClass   textclass;
  Coref       coref;

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
  static const char * const TAG_CONT;   //sent, word
  static const char * const TAG_POS;
  static const char * const TAG_NE;
  static const char * const TAG_WSD;
  static const char * const TAG_WSD_EXP;
  static const char * const TAG_PSR_PARENT;
  static const char * const TAG_PSR_RELATE;
  static const char * const TAG_SRL_ARG;
  static const char * const TAG_SRL_TYPE;
  static const char * const TAG_BEGIN;  // cr, srl
  static const char * const TAG_END;    // cr, srl
  static const char * const TAG_ID;   // para, sent, word
};

#endif    //  end for __LTP_XML4NLP_H__