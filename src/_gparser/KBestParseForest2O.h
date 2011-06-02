#ifndef _K_BEST_PARSE_FOREST_2O_
#define _K_BEST_PARSE_FOREST_2O_

#include "KBestParseForest.h"

class KBestParseForest2O : public KBestParseForest
{
public:
	KBestParseForest2O() {}

	KBestParseForest2O &reset(int _start, int _end, DepInstance *pInstance, int _K) {
		K = _K;
		start = _start;
		end = _end;
		sent = &(pInstance->forms);
		pos = &(pInstance->postags); 
		vector<unsigned int> chart_dim;
		chart.setDemisionVal(chart_dim, end+1, end+1, 2, 3, K);
		chart.resize(chart_dim);
		return *this;
	}

	~KBestParseForest2O(void) {}

	void getDepString(const ParseForestItem &pfi, string &strDep);

	void viterbi(	DepInstance *inst,
		MultiArray<FeatureVec> &fvs, 
		MultiArray<double>  &probs, 	
		MultiArray<FeatureVec> &fvs_trips,
		MultiArray<double> &probs_trips,
		MultiArray<FeatureVec> &fvs_sibs,
		MultiArray<double> &probs_sibs,
		MultiArray<FeatureVec> &nt_fvs,
		MultiArray<double> &nt_probs,	
		const MultiArray<int> &static_types, bool isLabeled);
};

#endif

