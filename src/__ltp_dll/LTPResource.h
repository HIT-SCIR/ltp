#ifndef _LTP_RESOURCE_H_
#define _LTP_RESOURCE_H_

#pragma warning(disable: 4786 4284)
#include <string>
using namespace std;

extern ofstream ltp_log_file;

#include "../__util/MyLib.h"
#include "../__xml4nlp/Xml4nlp.h"
#include "../_split_sentence/SplitSentence.h"
#include "../_crfwordsegment/CRFWS_DLL.h"
#pragma comment(lib, "_crfwordsegment.lib")

#include "../_svmtagger/svmtagger_dll.h"
#include "../_ner/NER_DLL.h"
#pragma comment(lib, "_ner.lib")
/*
#include "../_wsd/WSD_dll.h"
#pragma comment(lib, "_wsd.lib")
*/
#include "../_srl/SRL_DLL.h"
#pragma comment(lib, "_srl.lib")
#include "../_gparser/gparser_dll.h"
#pragma comment(lib, "_gparser.lib")

/*
#ifdef _WIN32

#include "../_irlas/IRLAS_DLL.h"

#include "../_parser/Parser_dll.h"
#pragma comment(lib, "_parser.lib")

#endif
*/

class LTPResource
{
public:
	LTPResource();
	~LTPResource();

	int LoadCRFWSRes(const char *dataFolder);
	int LoadCRFWSRes(const string &dataFolder) 
	{
		return LoadCRFWSRes( dataFolder.c_str() );
	}
	
	/*
	int LoadIrlasRes(const char *confFile, const char *dataFolder);
	int LoadIrlasRes(const string &confFile, const string &dataFolder) 
	{
		return LoadIrlasRes(confFile.c_str(), dataFolder.c_str());
	}
	*/
	int LoadSvmtaggerRes(const char *dataFolder);
	int LoadSvmtaggerRes(const string &dataFolder) 
	{
		return LoadSvmtaggerRes(dataFolder.c_str());
	}
	int LoadNeRes(const char *dataFolder);
	int LoadNeRes(const string &dataFolder) 
	{
		return LoadNeRes(dataFolder.c_str());
	}
	/*
	int LoadWsdRes(const char *dataFolder);
	int LoadWsdRes(const string &dataFolder)
	{
		return LoadWsdRes(dataFolder.c_str());
	}
	*/

	int LoadGParserRes(const char *dataFolder);
	int LoadGParserRes(const string &dataFolder)
	{
		return LoadGParserRes(dataFolder.c_str());
	}

	/*
	int LoadParserRes(const char *dataFolder);
	int LoadParserRes(const string &dataFolder)
	{
		return LoadParserRes(dataFolder.c_str());
	}
	*/

	int LoadSrlRes(const char *dataFolder);
	int LoadSrlRes(const string &dataFolder) 
	{
		return LoadSrlRes(dataFolder.c_str());
	}


	void ReleaseCRFWSRes();
	//int ReleaseIrlasRes();
	int ReleaseSvmtaggerRes();
	int ReleaseNeRes();
	//int ReleaseWsdRes();
	int ReleaseGParserRes();
	//int ReleaseParserRes();
	int ReleaseSrlRes();
	
	//void *GetIrlasSeggerPtr() { return m_irlasSeggerPtr; }
	void *GetNerPtr() {return m_nerPtr; }
	void *GetGParserPtr() {return m_gparserPtr; }
private:
	void *m_irlasSeggerPtr;
	void *m_nerPtr;
	void *m_gparserPtr;
private:
	// copy operator and assign operator is not allowed.

private:
	bool m_isLoadCRFWSRes;
	//bool m_isLoadIrlasRes;
	bool m_isLoadSvmtaggerRes;
	bool m_isLoadNeRes;
	//bool m_isLoadWsdRes;
	bool m_isLoadGParserRes;
	//bool m_isLoadParserRes;
	bool m_isLoadSrlRes;
};

#endif
