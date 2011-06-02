#include "LTPResource.h"

LTPResource::LTPResource() : 
				m_nerPtr(NULL),
				m_isLoadSvmtaggerRes(false), m_isLoadNeRes(false), m_isLoadGParserRes(false),
				m_isLoadSrlRes(false), m_isLoadCRFWSRes(false)
{
}

LTPResource::~LTPResource()
{
	ReleaseCRFWSRes();
	//ReleaseIrlasRes();
	ReleaseSvmtaggerRes();
	//ReleaseNeRes();
	//ReleaseWsdRes();
	//ReleaseGParserRes();
	//ReleaseParserRes();
	//ReleaseSrlRes();
}


/************* Load Resource ************/
/************* Release Resource ************/


int LTPResource::LoadCRFWSRes(const char *dataFolder)
{
	if (m_isLoadCRFWSRes) return 0;

	cerr << "load crf-ws resource from: " << dataFolder << endl;
	print_time();
	int ret = 0;
	try {
		ret = CRFWS_LoadResource(dataFolder);
	} catch (const exception &e) {
		ret = -2;
		cerr << e.what() << endl;
	}

	if (ret != 0) {
		cerr << "load crf-ws resource error!" << endl;
		return -1;
	}

	cerr << "load crf-ws resource over." << endl;
	print_time();

	m_isLoadCRFWSRes = true;
	return 0;
}


void LTPResource::ReleaseCRFWSRes()
{
	if (!m_isLoadCRFWSRes) return ;
	cerr << "start to release crf-ws resource..." << endl;
	print_time();

	CRFWS_ReleaseResource();
	cerr << "release crf-ws resource over" << endl;
	print_time();
	m_isLoadCRFWSRes = false;
}

int LTPResource::LoadSvmtaggerRes(const char *dataFolder)
{
	if (m_isLoadSvmtaggerRes) return 0;

	cerr << "load svmtagger resource from " << dataFolder << " ..." << endl;
	print_time();
	if (0 != svmtagger_LoadResource(dataFolder)) return -1;
	cerr << "load svmtagger resource over" << endl;
	print_time();

	m_isLoadSvmtaggerRes = true;
	return 0;
}

int LTPResource::ReleaseSvmtaggerRes()
{
	if (!m_isLoadSvmtaggerRes) return 0;

	cerr << "start to release svmtagger resource ..." << endl;
	print_time();
	svmtagger_ReleaseResource();
	cerr << "release svmtagger resource over" << endl;
	print_time();

	m_isLoadSvmtaggerRes = false;
	return 0;
}
int LTPResource::LoadNeRes(const char *dataFolder)
{
	if (m_isLoadNeRes) return 0;

	cerr << "load ner resource..." << endl;
	print_time();
	if ( !NER_LoadResource(const_cast<char *>(dataFolder)) )  //加载资源
	{
		cerr << "load ner resource error!" << endl;
		return -1;
	}	
	m_nerPtr = NER_CreateNErecoger();		//创建NE识别器

	cerr << "load ner resource over." << endl;
	print_time();
	
	m_isLoadNeRes = true;
	return 0;
}



int LTPResource::ReleaseNeRes()
{
	if (!m_isLoadNeRes) return 0;

	cerr << "start to release ner resource ..." << endl;
	print_time();

	NER_ReleaseNErecoger(m_nerPtr); //销毁NE识别器
	NER_ReleaseResource();	    //释放资源
	m_nerPtr = NULL;

	cerr << "release ner resource over" << endl;
	print_time();

	m_isLoadNeRes = false;
	return 0;
}

/*
int LTPResource::LoadWsdRes(const char *dataFolder)
{
	if (m_isLoadWsdRes) return 0;

	cerr << "load wsd resource... " << dataFolder << endl;
	print_time();
	WSD_LoadResource(dataFolder);
	cerr << "load wsd resource over" << endl;
	print_time();

	m_isLoadWsdRes = true;
	return 0;
}

int LTPResource::ReleaseWsdRes()
{
	if (!m_isLoadWsdRes) return 0;

	cerr << "start to release wsd resource ..." << endl;
	print_time();
	WSD_ReleaseResource();
	cerr << "release wsd resource over" << endl;
	print_time();

	m_isLoadWsdRes = false;
	return 0;
}
*/

int LTPResource::LoadGParserRes(const char *dataFolder)
{
	if (m_isLoadGParserRes) return 0;

	cerr << "load gparser resource..." << endl;
	print_time();

	string strDataFolder = dataFolder;
	strDataFolder += "config.ircdt_10k.txt";
	m_gparserPtr = GParser_CreateParser(strDataFolder.c_str());
	if (!m_gparserPtr) return -1;

	if (0 != GParser_LoadResource(m_gparserPtr, dataFolder)) return -1;

	cerr << "load gparser resource over" << endl;
	print_time();

	m_isLoadGParserRes = true;
	return 0;
}

int LTPResource::ReleaseGParserRes()
{
	if (!m_isLoadGParserRes) return 0;

	cerr << "start to release gparser resource ..." << endl;
	print_time();
	
	GParser_ReleaseResource(m_gparserPtr);
	GParser_ReleaseParser(m_gparserPtr);

	cerr << "release gparser resource over" << endl;
	print_time();

	m_isLoadGParserRes = false;
	return 0;
}

int LTPResource::LoadSrlRes(const char *dataFolder)
{
	if (m_isLoadSrlRes) return 0;

	cerr << "load srl resource..." << endl;
	print_time();
	if (0 != SRL_LoadResource(string(dataFolder))) return -1;
	cerr << "load srl resource over" << endl;
	print_time();

	m_isLoadSrlRes = true;
	return 0;
}

int LTPResource::ReleaseSrlRes()
{
	if (!m_isLoadSrlRes) return 0;

	cerr << "start to release srl resource ..." << endl;
	print_time();
	if (0 != SRL_ReleaseResource()) return -1;
	cerr << "release srl resource over" << endl;
	print_time();

	m_isLoadSrlRes = false;
	return 0;
}

#ifdef _WIN32

/*
int LTPResource::LoadIrlasRes(const char *confFile, const char *dataFolder)
{
if (m_isLoadIrlasRes) return 0;

cerr << "load irlas resource..." << endl;
print_time();
if (1 != IRLAS_LoadResource(confFile, dataFolder))
{
cerr << "load irlas resource error!" << endl;
return -1;
}
m_irlasSeggerPtr = IRLAS_CreateSegger();

cerr << "load irlas resource over." << endl;
print_time();

m_isLoadIrlasRes = true;
return 0;
}


int LTPResource::ReleaseIrlasRes()
{
if (!m_isLoadIrlasRes) return 0;
cerr << "start to release irlas resource..." << endl;
print_time();

IRLAS_ReleaseSegger(m_irlasSeggerPtr);
IRLAS_ReleaseResource();
m_irlasSeggerPtr = NULL;

cerr << "release irlas resource over" << endl;
print_time();
m_isLoadIrlasRes = false;
return 0;
}
*/

/*
int LTPResource::LoadParserRes(const char *dataFolder)
{
if (m_isLoadParserRes) return 0;

cerr << "load parser resource..." << endl;
print_time();
Parser_LoadResource(dataFolder);
cerr << "load parser resource over" << endl;

m_isLoadParserRes = true;
print_time();
return 0;
}

int LTPResource::ReleaseParserRes()
{
if (!m_isLoadParserRes) return 0;

cerr << "start to release parser resource ..." << endl;
print_time();
Parser_ReleaseResource();
cerr << "release parser resource over" << endl;
print_time();

m_isLoadParserRes = false;
return 0;
}
*/
#endif