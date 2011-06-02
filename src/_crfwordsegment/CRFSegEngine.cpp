#include "CRFSegEngine.h"
#include <sstream>
#include <util/EncodeUtil.h>
#include <util/TextProcess.h>

using namespace std;

LAS_NS_BEG

static const CRFPP::Option long_options[] = {
	{"model",  'm',  0,       "FILE",  "set FILE for model file"},
	{"nbest",  'n', "0",      "INT",   "output n-best results"},
	{"verbose" , 'v', "0",    "INT",   "set INT for verbose level"},
	{"cost-factor", 'c', "1.0", "FLOAT", "set cost factor"},
	{"output",         'o',  0,       "FILE",  "use FILE as output file"},
	{"version",        'v',  0,        0,       "show the version and exit" },
	{"help",   'h',  0,        0,       "show this help and exit" },
	{0, 0, 0, 0, 0}
};

CRFSegEngine::CRFSegEngine(const char *model_name)
{
	string opt("-m ");
	opt+=model_name;
	param.open(opt.c_str(), long_options);
	if( !tagger.open(&param) ) 
	{
		throw string( tagger.what() );
	}
	EncodeUtil::InitGbkU16();
}

CRFSegEngine::~CRFSegEngine(void)
{
}

void CRFSegEngine::Gbk2Utf8(const string from, string &to)
{
	wstring utf16, gbk;
	EncodeUtil::MultiByteToWideChar(from, gbk);
	EncodeUtil::GbkToUnicode(gbk, utf16);
	EncodeUtil::Utf16ToUtf8(utf16, to);
}

void CRFSegEngine::Utf82Gbk(const string from, string &to)
{
	wstring utf16, gbk;
	EncodeUtil::Utf8ToUtf16(from, utf16);
	EncodeUtil::UnicodeToGbk(utf16, gbk);
	EncodeUtil::WideCharToMultiByte(gbk, to);
}

// both the inputs and outputs are UTF8 encoded.
bool CRFSegEngine::Segment(const string &text, DictBase *pDict, std::vector<std::string> &vecSegResult)
{
	vecSegResult.clear();
	tagger.parse_stream(text, vecSegResult);
	return true;
}

// the input is UTF8 encoded
bool CRFSegEngine::Segment(const wstring &text, std::vector<std::pair<int, int> > &vecSegPos)
{
	// not available...
//	string utf8;
//	EncodeUtil::Utf16ToUtf8(text, utf8);
//	tagger.parse_stream(utf8, vecSegResult);
	return false;
}

// both inputs are GBK encoded
bool CRFSegEngine::Segment(const char* text, DictBase *pDict, std::vector<std::string> &vecSegResult)
{
	string utf8;
	Gbk2Utf8(string(text), utf8);

	vecSegResult.clear();
	tagger.parse_stream(utf8, vecSegResult);
	for(vector<string>::iterator i=vecSegResult.begin(); i!=vecSegResult.end(); ++i)
		Utf82Gbk(*i, *i);
	return true;
}

LAS_NS_END
