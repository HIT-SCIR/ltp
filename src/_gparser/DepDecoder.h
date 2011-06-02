#ifndef _DEP_DECODER_
#define _DEP_DECODER_

#pragma once
#include "DepPipe.h"
#include "ParseForestItem.h"
#include "KBestParseForest.h"
#include "FeatureVec.h"
#include "MultiArray.h"

#include <vector>
using namespace std;

/*
	this class implements Eisner algorithm for 1-order parsing 
*/
class DepDecoder
{
protected:
	const ParserOptions &options;
	DepPipe &pipe;
private:
	KBestParseForest pf;
public:
	DepDecoder(const ParserOptions &_options, DepPipe &_pipe) : options(_options), pipe(_pipe) {}
	virtual ~DepDecoder(void) {}

	void getTypes(const MultiArray<double> &nt_probs, int len, MultiArray<int> &type);

	// static type for each edge: run time O(n^3 + Tn^2) T is number of types
	virtual void decodeProjective(DepInstance *inst,
						  MultiArray<FeatureVec> &fvs,
						  MultiArray<double>  &probs,
						  MultiArray<FeatureVec> &fvs_trips,
						  MultiArray<double> &probs_trips,
						  MultiArray<FeatureVec> &fvs_sibs,
						  MultiArray<double> &probs_sibs,
						  MultiArray<FeatureVec> &nt_fvs,
						  MultiArray<double> &nt_probs,
						  int K,
						  vector<FeatureVec> &d0,
						  vector<string> &d1,
						  vector<double> &parse_probs)

	{
		MultiArray<int> static_types;
		if(options.m_isLabeled) {
			getTypes(nt_probs, inst->size(), static_types);
		}

		pf.reset(0, inst->size()-1, inst, K);
		pf.viterbi(inst, fvs, probs, nt_fvs, nt_probs, static_types, options.m_isLabeled);

		pf.getBestParses(d0, d1, parse_probs);
	}
};

#endif

