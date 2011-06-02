#ifndef _DEP_PIPE_2O_H_
#define _DEP_PIPE_2O_H_

#include "DepPipe.h"

class DepPipe2O : public DepPipe
{
public:
	DepPipe2O(const ParserOptions &_options);
	virtual ~DepPipe2O(void);

public:
	int readInstance(	FILE *featFile, int length,
		MultiArray<FeatureVec> &fvs,
		MultiArray<double> &probs,
		MultiArray<FeatureVec> &fvs_trips,
		MultiArray<double> &probs_trips,
		MultiArray<FeatureVec> &fvs_sibs,
		MultiArray<double> &probs_sibs,
		MultiArray<FeatureVec> &nt_fvs,
		MultiArray<double> &nt_probs,
		FeatureVec &fv,
		string &actParseTree,
		const Parameter &params);

	void fillFeatureVectors(DepInstance *instance,
		MultiArray<FeatureVec> &fvs,
		MultiArray<double> &probs,
		MultiArray<FeatureVec> &fvs_trips,
		MultiArray<double> &probs_trips,
		MultiArray<FeatureVec> &fvs_sibs,
		MultiArray<double> &probs_sibs,
		MultiArray<FeatureVec> &nt_fvs,
		MultiArray<double> &nt_probs,
		const Parameter &params);

protected:
	void addExtendedFeature(DepInstance *pInstance, FeatureVec &fv);
	void writeExtendedFeatures(DepInstance *pInstance, FILE *featFile);
	void addSibFeature(DepInstance *pInstance, int ch1, int ch2, bool isST, FeatureVec &fv);
	void addTripFeature(DepInstance *pInstance, int par, int ch1, int ch2, FeatureVec &fv);

};

#endif

