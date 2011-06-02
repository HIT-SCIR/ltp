#ifndef _DEP_DECODER_2O_
#define _DEP_DECODER_2O_

#include "DepDecoder.h"
#include "KBestParseForest2O.h"

class DepDecoder2O : public DepDecoder
{
public:
	DepDecoder2O(const ParserOptions &_options, DepPipe &_pipe) : DepDecoder(_options, _pipe) {}
	~DepDecoder2O(void) {}
private:
	KBestParseForest2O pf;

public:
	void decodeProjective(DepInstance *inst,
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
		pf.viterbi(inst, fvs, probs, fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, static_types, options.m_isLabeled);

		pf.getBestParses(d0, d1, parse_probs);
	}
};

#endif


