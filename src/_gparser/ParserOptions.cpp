#include "ParserOptions.h"
#include "MyLib.h"

ParserOptions::ParserOptions()
{
	m_isTrain = false;
	m_strTrainFile = "";
	m_strTrainForestFile = "";
	m_isTrainForestExists = false;
	m_numIter = 10;
	m_trainK = 5;
	m_strTrain_IterNums_to_SaveParamModel = "";
	m_setTrain_IterNums_to_SaveParamModel.clear();

	m_isTest = false;
	m_strTestFile = "";
	m_strOutFile = "";
	m_testK = 0;
	m_strTest_IterNum_of_ParamModel = "";

	m_isOutPutScore = false;

	m_isSecondOrder = false;
	m_isLabeled = false;
	m_strModelName = "default.model";
	m_numMaxInstance = -1;
	m_isCONLLFormat = true;

	m_display_interval = 1;

	m_isUseForm = false;
	m_isUseLemma = false;
	m_isUsePostag = false;
	m_isUseCPostag = false;
	m_isUseFeats = false;

	m_isUseForm_label = false;
	m_isUseLemma_label = false;

	m_isUse_label_feats_t_child = false;
	m_isUse_label_feats_t = false;

	m_isUse_arc_bet_each = false;
	m_isUse_arc_bet_same_num = false;


//	m_isEval = false;
//	m_strGoldFile = "";
}

ParserOptions::~ParserOptions()
{
}

int ParserOptions::setOptions(const char *option_file)
{
	cerr << "\ngparser-option-config file: " << option_file << endl;

	ifstream conf(option_file);
	if (!conf) return -1;

	vector<string> vecOpt;
	string strLine;
	while (my_getline(conf, strLine)) vecOpt.push_back(strLine);
	conf.close();

	setOptions(vecOpt);
	return 0;
}

void ParserOptions::setOptions(const vector<string> &vecOption)
{
	int i = 0;
	for (; i < vecOption.size(); ++i) {
		pair<string, string> pr;
		string2pair(vecOption[i], pr, ':');

		if (pr.first == "train") m_isTrain = true;
		if (pr.first == "train-file") m_strTrainFile = pr.second;
		if (pr.first == "train-forest-file") m_strTrainForestFile = pr.second;
		if (pr.first == "train-forest-exists") m_isTrainForestExists = atoi(pr.second.c_str()) == 0 ? false : true;
		if (pr.first == "iters") m_numIter = atoi(pr.second.c_str());
		if (pr.first == "train-k") m_trainK = atoi(pr.second.c_str());
		if (pr.first == "train-iter-nums-to-save-param-model") {
			m_strTrain_IterNums_to_SaveParamModel = pr.second;
			vector<string> vec;
			split_bychar(m_strTrain_IterNums_to_SaveParamModel, vec, '_');
			int j = 0;
			for (; j < vec.size(); ++j) {
				m_setTrain_IterNums_to_SaveParamModel.insert( atoi( vec[j].c_str() ) );
			}
		}

		if (pr.first == "test") m_isTest = true;
		if (pr.first == "test-file") m_strTestFile = pr.second;
		if (pr.first == "output-file") m_strOutFile = pr.second;
		if (pr.first == "test-k") m_testK = atoi(pr.second.c_str());
		if (pr.first == "test-param-model-iter-num") m_strTest_IterNum_of_ParamModel = pr.second;
		if (pr.first == "test-output-score") m_isOutPutScore = true;

		if (pr.first == "second-order") m_isSecondOrder = true;
		if (pr.first == "labeled") m_isLabeled = true;
		if (pr.first == "model-name") m_strModelName = pr.second;
		if (pr.first == "max-instance") m_numMaxInstance = atoi(pr.second.c_str());
		if (pr.first == "data-format") {
			if (pr.second == "conll") {
				m_isCONLLFormat = true;
			} else {
				m_isCONLLFormat = false;
			}
		}
		if (pr.first == "display-interval") m_display_interval = atoi(pr.second.c_str());

		if (pr.first == "use-form") m_isUseForm = true;
		if (pr.first == "use-lemma") m_isUseLemma = true;
		if (pr.first == "use-postag") m_isUsePostag = true;
		if (pr.first == "use-cpostag") m_isUseCPostag = true;
		if (pr.first == "use-feats") m_isUseFeats = true;

		if (pr.first == "use-form-label") m_isUseForm_label = true;
		if (pr.first == "use-lemma-label") m_isUseLemma_label = true;

		if (pr.first == "use-label-feats_t_child") m_isUse_label_feats_t_child = true;
		if (pr.first == "use-label-feats_t") m_isUse_label_feats_t = true;

		if (pr.first == "use-arc-bet-each") m_isUse_arc_bet_each = true;
		if (pr.first == "use-arc-bet-same-num") m_isUse_arc_bet_same_num = true;

//		if (pr.first == "eval") m_isEval = true;
//		if (pr.first == "gold-file") m_strGoldFile = pr.second;
	}
}


void ParserOptions::showOptions()
{
	cerr << "\n/*******configuration-beg*******/" << endl;

	if (m_isTrain) {
		cerr << ">train: " << endl;
		cerr << "\t" << "train-file: " << m_strTrainFile << endl;
		cerr << "\t" << "train-forest-file: " << m_strTrainForestFile << endl;
		cerr << "\t" << "train-forest-exist: " << (m_isTrainForestExists ? "yes" : "no") << endl;
		cerr << "\t" << "iteration-num: " << m_numIter << endl;
		cerr << "\t" << "train-k: " << m_trainK << endl;
		cerr << "\t" << "train-iter-nums-to-save-param-model: " << (m_strTrain_IterNums_to_SaveParamModel.empty() ? "not defined" : m_strTrain_IterNums_to_SaveParamModel) << endl;
	}

	if (m_isTest) {
		cerr << ">test: " << endl;
		cerr << "\t" << "test-file: " << m_strTestFile << endl;
		cerr << "\t" << "out-file: " << m_strOutFile << endl;
		cerr << "\t" << "test-k: " << m_testK << endl;
		cerr << "\t" << "test-param-model-iter-num: " << (m_strTest_IterNum_of_ParamModel.empty() ? "not defined" : m_strTest_IterNum_of_ParamModel) << endl;
		cerr << "\t" << "test-output-score: " << (m_isOutPutScore ? "yes" : "no") << endl;
	}

	cerr << ">other: " << endl;
	cerr << "\t" << "second-order: " << (m_isSecondOrder ? "yes" : "no") << endl;
	cerr << "\t" << "labeled: " << (m_isLabeled ? "yes" : "no") << endl;
	cerr << "\t" << "model-name: " << m_strModelName << endl;
	cerr << endl;

	cerr << "\t" << "instance-limit: " << m_numMaxInstance << endl;
	cerr << "\t" << "data-format: " << (m_isCONLLFormat ? "conll" : "not-conll") << endl;
	cerr << "\t" << "display-interval: " << m_display_interval << endl;

	cerr << ">features: " << endl;
	cerr << "\t" << "use-form: " << (m_isUseForm ? "yes" : "no") << endl;
	cerr << "\t" << "use-lemma: " << (m_isUseLemma ? "yes" : "no") << endl;
	cerr << "\t" << "use-postag: " << (m_isUsePostag ? "yes" : "no") << endl;
	cerr << "\t" << "use-cpostag: " << (m_isUseCPostag ? "yes" : "no") << endl;
	cerr << "\t" << "use-feats: " << (m_isUseFeats ? "yes" : "no") << endl;
	cerr << endl;
	cerr << "\t" << "use-form-label: " << (m_isUseForm_label ? "yes" : "no") << endl;
	cerr << "\t" << "use-lemma-label: " << (m_isUseLemma_label ? "yes" : "no") << endl;
	cerr << "\t" << "use-label-feats_t_child [such as form_cpos_type_is-child]: " << (m_isUse_label_feats_t_child ? "yes" : "no") << endl;
	cerr << "\t" << "use-label-feats_t [such as form_cpos_type]: " << (m_isUse_label_feats_t ? "yes" : "no") << endl;

	cerr << endl;
	cerr << "\t" << "use-arc-bet-each: " << (m_isUse_arc_bet_each ? "yes" : "no") << endl;
	cerr << "\t" << "use-arc-bet-same-num: " << (m_isUse_arc_bet_same_num ? "yes" : "no") << endl;

	cerr << "/*******configuration-end*******/" << endl;
}


