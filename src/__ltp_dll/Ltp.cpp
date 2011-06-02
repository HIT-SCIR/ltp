#include "Ltp.h"
#include <ctime>
#include <map>
#include <string>
using namespace std;

const unsigned int LTP::DO_XML = 1;
const unsigned int LTP::DO_SPLITSENTENCE = 1 << 1;
//const unsigned int LTP::DO_IRLAS = 1 << 2;
const unsigned int LTP::DO_NER = 1 << 3;
const unsigned int LTP::DO_PARSER = 1 << 4;
//const unsigned int LTP::DO_WSD = 1 << 5;
const unsigned int LTP::DO_SRL = 1 << 6;


LTP::LTP(XML4NLP &xml4nlp) : m_ltpResource(), m_ltpOption(), m_xml4nlp(xml4nlp)
{
	ReadConfFile();
}

LTP::~LTP()
{
}

int LTP::ReadConfFile(const char *confFileName)
{
	ifstream confFile(confFileName);
	if (!confFile.is_open())
	{
		cerr << "open ltp config file err: " << confFileName << endl;
		return -1;
	}
	
	map<string, string> mapConf;
	string strLine;
	while (getline(confFile, strLine))
	{
		if (strLine.empty()) continue;
		if( strLine.at(0) == '#') continue;
		size_t pos = strLine.find_first_of("# \t");
		if (pos != string::npos)
			strLine = strLine.substr(0, pos);
		pos = strLine.find_first_of("=");
		if (pos == string::npos) continue;
		string strKey = strLine.substr(0, pos);
		string strValue = strLine.substr(pos+1);
        mapConf[strKey] = strValue;		
	}

	confFile.close();
	
	map<string, string>::const_iterator it = mapConf.find("LTP_dataFolder");
	if (it == mapConf.end() || it->second.empty())
		m_ltpOption.strLTPDataPath = "./ltp_data/";
	else
		m_ltpOption.strLTPDataPath = it->second;

	// IRLAS conf
	/*
	it = mapConf.find("IRLAS_confFile");
	if (it == mapConf.end() || it->second.empty())
		m_ltpOption.irlasOpt.confFile = "./ltp_data/irlas_data/irlas_config.ini";
	else
		m_ltpOption.irlasOpt.confFile = it->second;
	*/
	
	// NE conf, ONLY use default
	it = mapConf.find("NE_Entity");
	if (it == mapConf.end() || it->second.empty())
		m_ltpOption.neOpt.isEntity = 1;
	else
		m_ltpOption.neOpt.isEntity = atoi( it->second.c_str() );
	
	it = mapConf.find("NE_Time");
	if (it == mapConf.end() || it->second.empty())
		m_ltpOption.neOpt.isTime = 1;
	else
		m_ltpOption.neOpt.isTime = atoi( it->second.c_str() );
	
	it = mapConf.find("NE_Num");	
	if (it == mapConf.end() || it->second.empty())
		m_ltpOption.neOpt.isNum = 1;
	else
		m_ltpOption.neOpt.isNum = atoi( it->second.c_str() );
	
	//map<string, string>::const_iterator it = mapConf.begin();
	//for (; it != mapConf.end(); ++it)
	//	cout << it->first << " " << it->second << endl;

	//m_ltpOption.output();
	return 0;
}

// If you do NOT split sentence explicitly, this will be called according to dependencies among modules
int LTP::splitSentence_dummy()
{
	if ( m_xml4nlp.QueryNote(NOTE_SENT) ) return 0;
	int paraNum = m_xml4nlp.CountParagraphInDocument();

	if (paraNum == 0)
	{
		cerr << "There is no paragraph in doc," << endl
			<< "you may have loaded a blank file or have not loaded a file yet" << endl;
		return -1;
	}

	for (int i = 0; i < paraNum; ++i)
	{
		vector<string> vecSentences;
		string para;
		m_xml4nlp.GetParagraph(i, para);
		if (0 == SplitSentence( para, vecSentences )) return -1; // func SplitSentence's return val is vecSentence.size() 
		// dummy
		// vecSentences.push_back(para);
		if (0 != m_xml4nlp.SetSentencesToParagraph(vecSentences, i)) return -1;
	}

	m_xml4nlp.SetNote(NOTE_SENT);
	return 0;
}

int LTP::crfWordSeg()
{
	if ( m_xml4nlp.QueryNote(NOTE_WORD) ) return 0;

	// dependency
	if (0 != splitSentence_dummy()) return -1;

	if (0 != m_ltpResource.LoadCRFWSRes(m_ltpOption.strLTPDataPath+"crfws_data/")) return -1;

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when crfWordSeg" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		string strStn = m_xml4nlp.GetSentence(i);
		vector<string> vctWords;
		if (0 != CRFWS_WordSegment_x(strStn, vctWords)) return -1;
		if (m_xml4nlp.SetWordsToSentence(vctWords, i) != 0) return -1;
	}

	m_xml4nlp.SetNote(NOTE_WORD);

	return 0;
}

int LTP::postag()
{
	if ( m_xml4nlp.QueryNote(NOTE_POS) ) return 0;

	// dependency
	if (0 != crfWordSeg()) return -1;

	if (0 != m_ltpResource.LoadSvmtaggerRes(m_ltpOption.strLTPDataPath + "svmtagger_data/")) return -1;


	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when postagger()" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		vector<string> vecWord;
		m_xml4nlp.GetWordsFromSentence(vecWord, i);
//		copy(vecWord.begin(), vecWord.end(), ostream_iterator<string>(cout, "#"));
//		cout << endl;
		vector<string> vecPOS;
		svmtagger_PosTag(vecWord, vecPOS);
		if (m_xml4nlp.SetPOSsToSentence(vecPOS, i) != 0) return -1;
	}

	m_xml4nlp.SetNote(NOTE_POS);

	return 0;
}

int LTP::ner()
{
	if ( m_xml4nlp.QueryNote(NOTE_NE) ) return 0;

	// dependency
//	m_ltpOption.irlasOpt = IRLASOption(1, 1, 1);
//	if (0 != irlas()) return -1;
//	if (0 != segmentWord()) return -1;
	if (0 != postag()) return -1;

	if (0 != m_ltpResource.LoadNeRes(m_ltpOption.strLTPDataPath+"ne_data/")) return -1;
	void *pNer = m_ltpResource.GetNerPtr();		//创建NE识别器
	if (pNer == NULL) 
	{
		cerr << "pNer == NULL" << endl;
		return -1;
	}

	NER_SetOption(m_ltpOption.neOpt.isEntity, m_ltpOption.neOpt.isTime, m_ltpOption.neOpt.isNum);
	//cout << m_neOption.isEntity << "\t" << m_neOption.isTime << "\t" << m_neOption.isNum << endl;

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when ner" << endl;
		return -1;
	}
	for (int i=0; i<stnsNum; ++i)
	{
		vector<string> vecWord;
		vector<string> vecPOS;
		vector<string> vecNETag;
		if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) return -1;
		if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) return -1;
		if (0 != NER(pNer, vecWord, vecPOS, vecNETag)) return -1;
		//copy(vecNETags.begin(), vecNETags.end(), ostream_iterator<string>(cout, "\n"));
		//cout << vecNETags.size() << endl;
		m_xml4nlp.SetNEsToSentence(vecNETag, i);
	}
	m_xml4nlp.SetNote(NOTE_NE);

	return 0;
}

int LTP::gparser()
{
	if ( m_xml4nlp.QueryNote(NOTE_PARSER) ) return 0;

	// dependency
//	m_ltpOption.irlasOpt = IRLASOption(1, 1, 1);
//	if (0 != irlas()) return -1;
//	if (0 != segmentWord()) return -1;
	if (0 != postag()) return -1;

	if ( 0 != m_ltpResource.LoadGParserRes(m_ltpOption.strLTPDataPath + "gparser_data/") ) return -1;

	void* pParser = m_ltpResource.GetGParserPtr();
	if (pParser == NULL) 
	{
		cerr << "parserPtr == NULL" << endl;
	}

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when gparser" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		vector<string> vecWord;
		vector<string> vecPOS;
		vector<int> vecHead;
		vector<string> vecRel;
		vector< pair<int, string> > vecParse;
		if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) return -1;
		if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) return -1;

//		copy(vecWord.begin(), vecWord.end(), ostream_iterator<string>(cout, " "));
//		cout << endl;
//		copy(vecPOS.begin(), vecPOS.end(), ostream_iterator<string>(cout, " "));
//		cout << endl;
//		

		if (0 != GParser_Parse_x(pParser, vecWord, vecPOS, vecHead, vecRel)) return -1;
		
		if (0 != m_xml4nlp.SetParsesToSentence(vecHead, vecRel, i)) return -1;
	}

	m_xml4nlp.SetNote(NOTE_PARSER);

	return 0;
}

int LTP::srl()
{
	if ( m_xml4nlp.QueryNote(NOTE_SRL) ) return 0;

	// dependency
	if (0 != ner()) return -1;
	if (0 != gparser()) return -1;

	if ( 0 != m_ltpResource.LoadSrlRes(m_ltpOption.strLTPDataPath + "srl_data/") ) return -1;

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when srl" << endl;
		return -1;
	}

	for (int i = 0; i < stnsNum; ++i)
	{
		vector<string> vecWord;
		vector<string> vecPOS;
		vector<string> vecNE;
		vector< pair<int, string> > vecParse;
		vector< pair< int, vector< pair<const char *, pair< int, int > > > > > vecSRLResult;
		// cerr << "#" << m_xml4nlp.GetSentence(i) << "#\n" << endl;
		if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) return -1;
		if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) return -1;
		if (m_xml4nlp.GetNEsFromSentence(vecNE, i) != 0) return -1;
		if (m_xml4nlp.GetParsesFromSentence(vecParse, i) != 0) return -1;
		if (0 != SRL(vecWord, vecPOS, vecNE, vecParse, vecSRLResult)) {
			cerr << "srl err in sent: " << i << endl;
			cerr << m_xml4nlp.GetSentence(i) << endl;
			return -1;
		}
		int j = 0;
		for (; j < vecSRLResult.size(); ++j) {
			vector<string> vecType;
			vector< pair<int, int> > vecBegEnd;
			int k = 0;
			for (; k < vecSRLResult[j].second.size(); ++k) {
				vecType.push_back(vecSRLResult[j].second[k].first);
				vecBegEnd.push_back(vecSRLResult[j].second[k].second);
			}
			if (0 != m_xml4nlp.SetPredArgToWord(i, vecSRLResult[j].first, vecType, vecBegEnd)) return -1;
		}
	}

	m_xml4nlp.SetNote(NOTE_SRL);
	return 0;
}

/*
int LTP::wsd()
{
	if ( m_xml4nlp.QueryNote(NOTE_WSD) ) return 0;

	// dependency
//	m_ltpOption.irlasOpt = IRLASOption(1, 1, 0);
//	if (0 != irlas()) return -1;
	if (0 != crfWordSeg()) return -1;

	if (0 != m_ltpResource.LoadWsdRes(m_ltpOption.strLTPDataPath + "wsd_data/")) return -1;

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when wsd" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		vector<string> vecWord;
		vector<string> vecWSD;
		vector<string> vecExplain;
		vector< vector<string> > vecAll;
		if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) return -1;
		if (!WSD(vecWord, vecWSD, vecExplain, vecAll)) return -1;
		if (m_xml4nlp.SetWSDsToSentence(vecWSD, i) != 0) return -1;
		if (m_xml4nlp.SetWSDExplainsToSentence(vecExplain, i) != 0) return -1;
	}
	m_xml4nlp.SetNote(NOTE_WSD);

	return 0;
}
*/

#ifdef _WIN32

int LTP::main2(const char *cszFileName, const char *cszSaveFileName, unsigned int flag)
{	
	if (flag & DO_XML)
	{
		if (CreateDOMFromXml(cszFileName) != 0) return -11;
	}
	else
	{
		if (CreateDOMFromTxt(cszFileName) != 0) return -12;
	}

	if (flag & DO_SPLITSENTENCE) {
		if (splitSentence() != 0) return -1;
	}
	/*
	if (flag & DO_IRLAS) {
		if (postag() != 0) return -2;
	}
	*/
	if (flag & DO_NER) {
		if (ner() != 0) return -3;
	}
	/*
	if (flag & DO_WSD) {
		if (wsd() != 0) return -4;
	}
	*/
	if (flag & DO_PARSER) {
		if (gparser() != 0) return -5;
	}
	if (flag & DO_SRL) {
		if (srl() != 0) return -6;
	}
	if (SaveDOM(cszSaveFileName) != 0) return -10;

	return 0;
}

/*
int LTP::parser()
{
	if ( m_xml4nlp.QueryNote(NOTE_PARSER) ) return 0;

	// dependency
	//	m_ltpOption.irlasOpt = IRLASOption(1, 1, 1);
	//	if (0 != irlas()) return -1;
	//if (0 != segmentWord()) return -1;
	if (0 != postag()) return -1;

	if ( 0 != m_ltpResource.LoadParserRes(m_ltpOption.strLTPDataPath + "parser_data/") ) return -1;

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when parser" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		vector<string> vecWord;
		vector<string> vecPOS;
		vector< pair<int, string> > vecParse;
		if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) return -1;
		if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) return -1;
		//		copy(vecWord.begin(), vecWord.end(), ostream_iterator<string>(cout, " "));
		//cout << endl;
		//copy(vecPOS.begin(), vecPOS.end(), ostream_iterator<string>(cout, " "));
		//cout << endl;
		//		
		if (0 != Parser(vecWord, vecPOS, vecParse)) return -1;
		if (m_xml4nlp.SetParsesToSentence(vecParse, i) != 0) return -1;
	}
	m_xml4nlp.SetNote(NOTE_PARSER);

	return 0;
}
*/
/*
int LTP::segmentWord()
{
	if ( m_xml4nlp.QueryNote(NOTE_WORD) ) return 0;

	// dependency
	if (0 != splitSentence_dummy()) return -1;

	// Special Process, need reconsiderations...
	m_ltpOption.irlasOpt = IRLASOption(1, 1, 0);


	if (0 != m_ltpResource.LoadIrlasRes(m_ltpOption.irlasOpt.confFile, m_ltpOption.strLTPDataPath+"irlas_data/")) return -1;

	void* pSegger = m_ltpResource.GetIrlasSeggerPtr();
	if (pSegger == NULL)
	{
		cerr << "IrlasSeggerPtr == NULL" << endl;
		return -1;
	}

	IRLAS_SetOption(pSegger, m_ltpOption.irlasOpt.isPER, m_ltpOption.irlasOpt.isLOC, 0); // MUST NOT do postag!
 
	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when segmentWord" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		string strStns = m_xml4nlp.GetSentence(i);
		vector<string> vecWord;
		IRLAS(pSegger, strStns, vecWord);
		if (m_xml4nlp.SetWordsToSentence(vecWord, i) != 0) return -1;
	}

	m_xml4nlp.SetNote(NOTE_WORD);

	return 0;
}
*/
// Need to split sentence explicitly.
int LTP::splitSentence()
{
	if ( m_xml4nlp.QueryNote(NOTE_SENT) ) return 0;

	int paraNum = m_xml4nlp.CountParagraphInDocument();

	if (paraNum == 0)
	{
		cerr << "There is no paragraph in doc," << endl
			<< "you may have loaded a blank file or have not loaded a file yet" << endl;
		return -1;
	}

	for (int i = 0; i < paraNum; ++i)
	{
		vector<string> vecSentences;
		string para;
		m_xml4nlp.GetParagraph(i, para);
		if (0 == SplitSentence( para, vecSentences )) return -1; // func SplitSentence's return val is vecSentence.size() 
		//vecSentences.push_back(para);
		if (0 != m_xml4nlp.SetSentencesToParagraph(vecSentences, i)) return -1;
	}

	m_xml4nlp.SetNote(NOTE_SENT);
	return 0;
}

void SplitWordPOS(const vector<string> &vecWordPOS, vector<string> &vecWord,
				  vector<string> &vecPOS)
{
	vector< pair<string, string> > vecPair;
	convert_to_pair(vecWordPOS, vecPair);
	for (int i=0; i<vecPair.size(); ++i)
	{
		vecWord.push_back(vecPair[i].first);
		vecPOS.push_back(vecPair[i].second);
	}
}

/*
int LTP::irlas()
{	
	if (m_xml4nlp.QueryNote(NOTE_WORD)) return 0;

	// dependency
	if (0 != splitSentence_dummy()) return -1;

	if (0 != m_ltpResource.LoadIrlasRes(m_ltpOption.irlasOpt.confFile, m_ltpOption.strLTPDataPath+"irlas_data/")) return -1;

	void* pSegger = m_ltpResource.GetIrlasSeggerPtr();
	if (pSegger == NULL) 
	{
		cerr << "IrlasSeggerPtr == NULL" << endl;
		return -1;
	}

	IRLAS_SetOption(pSegger, m_ltpOption.irlasOpt.isPER, m_ltpOption.irlasOpt.isLOC, 1); // MUST do postag!

	int stnsNum = m_xml4nlp.CountSentenceInDocument();
	if (stnsNum == 0)
	{
		cerr << "stnsNum == 0 when irlas" << endl;
		return -1;
	}
	for (int i = 0; i < stnsNum; ++i)
	{
		string strStns = m_xml4nlp.GetSentence(i);
		vector<string> vecWordPOS;
		IRLAS(pSegger, strStns, vecWordPOS);
		// copy(vecWordPOS.begin(), vecWordPOS.end(), ostream_iterator<string>(cout, " "));
		vector<string> vecWord;
		vector<string> vecPOS;
		SplitWordPOS(vecWordPOS, vecWord, vecPOS);
		if (m_xml4nlp.SetWordsToSentence(vecWord, i) != 0) return -1;
		if (m_xml4nlp.SetPOSsToSentence(vecPOS, i) != 0) return -1;
	}

	m_xml4nlp.SetNote(NOTE_WORD);
	m_xml4nlp.SetNote(NOTE_POS);

	return 0;
}
*/
#endif
