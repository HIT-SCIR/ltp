#include "gparser_dll.h"
#include "DepPipe.h"
#include "ParserOptions.h"
#include "DepParser.h"
#include "MyLib.h"

class GParser {
public:
	ParserOptions *options;
	DepPipe *pipe;
	DepDecoder *decoder;
	DepParser *parser;

public:
	GParser() : options(0), pipe(0), decoder(0), parser(0) {}
	~GParser() {
		if (options) delete options;
		if (pipe) delete pipe;
		if (decoder) delete decoder;
		if (parser) delete parser;
	}
};

int GParser_Parse(void *gparser,
					const vector<string>& vecWord,
					const vector<string>& vecCPOS,
					char *szHeads,
					char *szLabels,
					int &nHeadsSize,
					int &nLablesSize)
{
	DepParser *parser = ((GParser *) gparser)->parser;
	vector<int> vecHead;
	vector<string> vecLabel;
	if (0 != parser->parseSent(vecWord, vecCPOS, vecHead, vecLabel)) {
		return -1;
	}

	// post process

	// avoid situation when word depends on punctuation.
	int i = 0;
	for (; i < vecHead.size(); ++i) {
		int headId = vecHead[i];
		int headIdx = headId - 1;
		while (1) {
			if (headId == 0) break;
			if (vecCPOS[headIdx] == "wp") {
				headId = vecHead[headIdx];
				headIdx = headId - 1;
			} else {
				break;
			}
		}
		vecHead[i] = headId;
	}

	// id -> idx
//	int is_root_found = 0;
	i = 0;
	for (; i < vecHead.size(); ++i) {
		--vecHead[i];
		if (vecCPOS[i] == "wp") {
			vecHead[i] = -2;
			//vecLabel[i] = "PUN";
		}

//		if (vecHead[i] == -1 && vecLabel[i] == "HED") {
//			is_root_found = 1;
//		}
	}

/*	if (!is_root_found) {
		if (!vecHead.empty()) {
			vecHead[0] = -1;
			vecLabel[0] = "HED";
		}
	}
*/
	vector<string> vecStrHead;
	int2str_vec(vecHead, vecStrHead);
	string strHeads;
	join_bystr(vecStrHead, strHeads, "\t");
	string strLabels;
	join_bystr(vecLabel, strLabels, "\t");

	if (nHeadsSize <= strHeads.size()) {
		nHeadsSize = strHeads.size() + 5;
		return -11;
	}

	if (nLablesSize <= strLabels.size()) {
		nLablesSize = strLabels.size() + 5;
		return -12;
	}

	strcpy(szHeads, strHeads.c_str());
	strcpy(szLabels, strLabels.c_str());
	return 0;
}

void *GParser_CreateParser(const char *szConfigFilePathName)
{
	GParser *gparser = new GParser();
	if (!gparser) return 0;

	// initialize
	gparser->options = new ParserOptions();
	if (0 == gparser->options) {
		cerr << "gparser new ParserOptions failed" << endl;
		goto ERR_CREATE_PARSER;
	}

	if (0 != gparser->options->setOptions(szConfigFilePathName)) goto ERR_CREATE_PARSER;
//	gparser->options->showOptions();

	if (gparser->options->m_isSecondOrder) {
		gparser->pipe = new DepPipe2O(*(gparser->options));
		gparser->decoder = new DepDecoder2O(*(gparser->options), *(gparser->pipe));
	} else {
		gparser->pipe = new DepPipe(*(gparser->options));
		gparser->decoder = new DepDecoder(*(gparser->options), *(gparser->pipe));
	}
	if (0 == gparser->pipe) {
		cerr << "gparser new DepPipe failed" << endl;
		goto ERR_CREATE_PARSER;
	}
	if (0 == gparser->decoder) {
		cerr << "gparser new DepDecoder failed" << endl;
		goto ERR_CREATE_PARSER;
	}

	gparser->parser = new DepParser(*(gparser->options), *(gparser->pipe), *(gparser->decoder));
	if (0 == gparser->parser) {
		cerr << "gparser new DepParser failed" << endl;
		goto ERR_CREATE_PARSER;
	}

	return (void *)gparser;

ERR_CREATE_PARSER:
	delete gparser;
	return 0;
}

int GParser_LoadResource(void *gparser, const char *szResourcePath)
{
	DepParser *parser = ((GParser *)gparser)->parser;
	ParserOptions *options = ((GParser *)gparser)->options;
	DepPipe *pipe = ((GParser *)gparser)->pipe;

	if ( 0 != parser->loadAlphabetModel(szResourcePath, options->m_strModelName.c_str()) ) {
		return -1;
	}
	if ( 0 != parser->loadParamModel(szResourcePath, options->m_strModelName.c_str(), options->m_strTest_IterNum_of_ParamModel.c_str()) ) {
		return -2;
	}

	pipe->closeAlphabet();
	return 0;
}

int GParser_ReleaseParser(void *&gparser)
{
	if (gparser) delete ((GParser *)gparser);
	gparser = 0;
	return 0;
}

int GParser_ReleaseResource(void *gparser)
{
	return 0;
}
