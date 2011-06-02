#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>
using namespace std;

/*
	this class implements global options for parser. include:
		2-order or 1-order
		prof or non-proj

		parameters:
			iter-num
			k-best
			feature-set
			...
*/

class ParserOptions
{
public:
	ParserOptions();
	int setOptions(const char *option_file);
	void setOptions(const vector<string> &vecOption);
	void showOptions();
	~ParserOptions();

public:
	bool m_isTrain;
	string m_strTrainFile;
	string m_strTrainForestFile;
	bool m_isTrainForestExists;
	int m_numIter;
	int m_trainK;
	string m_strTrain_IterNums_to_SaveParamModel;
	set<int> m_setTrain_IterNums_to_SaveParamModel;

	bool m_isTest;
	string m_strTestFile;
	string m_strOutFile;
	int m_testK;
	string m_strTest_IterNum_of_ParamModel;

	bool m_isOutPutScore;

	bool m_isSecondOrder;
	bool m_isLabeled;
	string m_strModelName;
	int m_numMaxInstance;
	bool m_isCONLLFormat;


	int m_display_interval;

	bool m_isUseForm;
	bool m_isUseLemma;
	bool m_isUsePostag;
	bool m_isUseCPostag;
	bool m_isUseFeats;
	
	bool m_isUseForm_label;
	bool m_isUseLemma_label;
	bool m_isUse_label_feats_t_child;
	bool m_isUse_label_feats_t;

	bool m_isUse_arc_bet_each;
	bool m_isUse_arc_bet_same_num;


//	bool m_isEval;
//	string m_strGoldFile;
};

#endif

