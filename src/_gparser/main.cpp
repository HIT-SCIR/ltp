#include "DepPipe.h"
#include "ParserOptions.h"
#include "DepParser.h"
#include "MyLib.h"
using namespace std;

int main(int argc, const char *argv[])
{
	string strConfigFileName;
	if (argc < 2) {
		cerr << "format: gparser option-config-file" << endl;
		strConfigFileName = "default_config.txt";
//		return -1;
	} else {
		strConfigFileName = argv[1];
	}

	ParserOptions options;
	options.setOptions(strConfigFileName.c_str());
	options.showOptions();

	if (options.m_isTrain) {
		cerr << "start training..." << endl;
		print_time();

		DepPipe *pipe = NULL;
		DepDecoder *decoder = NULL;
		if (options.m_isSecondOrder) {
			pipe = new DepPipe2O(options);
			decoder = new DepDecoder2O(options, *pipe);
		} else {
			pipe = new DepPipe(options);
			decoder = new DepDecoder(options, *pipe);
		}
		if (NULL == pipe) {
			cerr << "new DepPipe failed" << endl;
			exit(0);
		}
		if (NULL == decoder) {
			cerr << "new DepDecoder failed" << endl;
			exit(0);
		}

		vector<int> instanceLengths;
		pipe->createAlphabet(instanceLengths);
		pipe->closeAlphabet();
		
//		pipe.m_featAlphabet.show();
//		pipe.m_labelAlphabet.show();
//		exit(1);

		if (!options.m_strTrainForestFile.empty() && !options.m_isTrainForestExists) {
			pipe->createInstances();
		}

		DepParser dp(options, *pipe, *decoder);
		dp.saveAlphabetModel(options.m_strModelName.c_str());

		dp.train(instanceLengths);
		cerr << "training over" << endl;
		print_time();
		if (options.m_setTrain_IterNums_to_SaveParamModel.empty()) {
			dp.saveParamModel(options.m_strModelName.c_str(), "");
		}

		if (pipe) delete pipe;
		if (decoder) delete decoder;
	}

	if (options.m_isTest) {
		DepPipe *pipe = NULL;
		DepDecoder *decoder = NULL;
		if (options.m_isSecondOrder) {
			pipe = new DepPipe2O(options);
			decoder = new DepDecoder2O(options, *pipe);
		} else {
			pipe = new DepPipe(options);
			decoder = new DepDecoder(options, *pipe);
		}
		if (NULL == pipe) {
			cerr << "new DepPipe failed" << endl;
			exit(0);
		}
		if (NULL == decoder) {
			cerr << "new DepDecoder failed" << endl;
			exit(0);
		}

		DepParser dp(options, *pipe, *decoder);

		cerr << "Loading model..." << endl;
		print_time();
		if ( 0 != dp.loadAlphabetModel("./", options.m_strModelName.c_str()) ) {
			exit(0);
		}

		if ( 0 != dp.loadParamModel("./", options.m_strModelName.c_str(), options.m_strTest_IterNum_of_ParamModel.c_str()) ) {
			exit(0);
		}
		pipe->closeAlphabet();
		cerr << "done." << endl;
		print_time();

		dp.outputParses();
		print_time();
		if (pipe) delete pipe;
		if (decoder) delete decoder;
	}

	cerr << "\n-----\n" << endl;

//	if (options.m_isEval) {
//		cerr << "\nEVALUATION PERFORMANCE:" << endl;
//		DepEvaluator.evaluate(options.m_strGoldFile.c_str(), options.m_strOutFile.c_str());
//	}
	return 0;
}
