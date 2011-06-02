#ifndef _DEP_PARSER_
#define _DEP_PARSER_

#pragma once
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

#include "Parameter.h"
#include "DepDecoder.h"
#include "DepDecoder2O.h"
#include "ParserOptions.h"
#include "DepPipe.h"
#include "DepPipe2O.h"
#include "MyLib.h"

/*
	this class controls the parsing process.
*/

class DepParser
{
private:
	const ParserOptions &options;
	DepPipe &pipe;
	DepDecoder &decoder;
	Parameter params;

	// Get production crap.
	MultiArray<FeatureVec> fvs;
	MultiArray<double> probs;
	MultiArray<FeatureVec> fvs_trips;
	MultiArray<double> probs_trips;
	MultiArray<FeatureVec> fvs_sibs;
	MultiArray<double> probs_sibs;
	MultiArray<FeatureVec> nt_fvs;
	MultiArray<double> nt_probs;

public:
	DepParser(const ParserOptions &_options, DepPipe &_pipe, DepDecoder &_decoder) : options(_options), pipe(_pipe), decoder(_decoder), params(pipe.m_featAlphabet.size(), _options) {}
	~DepParser(void) {}

	void train(const vector<int> &instanceLengths);

//	void trainingIter(const vector<int> &instanceLengths, int iter);
	void trainingIter(const vector<int> &instanceLengths, int iter);

	//////////////////////////////////////////////////////
	// Get Best Parses ///////////////////////////////////
	//////////////////////////////////////////////////////
	void outputParses ();
	int parseSent(const vector<string> &vecWord,
				  const vector<string> &vecCPOS,
				  vector<int> &vecHead,
				  vector<string> &vecRel);


//	int saveModel(const char *modelName);
	int saveParamModel(const char *modelName, const char *paramModelIterNum);
	int saveAlphabetModel(const char *modelName);

//	int loadModel(const char *modelName);
	int loadParamModel(const char *modelPath, const char *modelName, const char *paramModelIterNum);
	int loadAlphabetModel(const char *modelPath, const char *modelName);

private:
	void fillInstance_k(DepInstance &inst, const vector<string> d1, const vector<double> &parse_probs);
	void fillParseResult(const string &tree_span, double prob, vector<int> &heads, vector<string> &deprels);
	void fillInstance(DepInstance &inst, const string &tree_span, double prob);

	int initInstance(DepInstance &inst,
		const vector<string> &vecWord,
		const vector<string> &vecCPOS);

	int getParseResult(const DepInstance &inst,
		vector<int> &vecHead,
		vector<string> &vecRel);

	// allocate space for all multi-array.
	int allocMultiArr(int length) {
		vector<unsigned int> fvs_dim;
		fvs.setDemisionVal(fvs_dim, length, length, 2);
		probs.resize(fvs_dim);
		if (0 > fvs.resize(fvs_dim)) return -1;
		if (options.m_isLabeled) {
			vector<unsigned int> nt_dim(4);
			nt_fvs.setDemisionVal(nt_dim, length, pipe.m_vecTypes.size(), 2, 2);
			nt_fvs.resize(nt_dim);
			nt_probs.resize(nt_dim);
		}
		if (options.m_isSecondOrder) {
			fvs_trips.setDemisionVal(fvs_dim, length, length, length);
			fvs_trips.resize(fvs_dim);
			probs_trips.resize(fvs_dim);

			fvs_sibs.setDemisionVal(fvs_dim, length, length, 2);
			fvs_sibs.resize(fvs_dim);
			probs_sibs.resize(fvs_dim);
		}
		return 0;
	}
};

#endif


