#ifndef _SRL_BASELINE_
#define _SRL_BASELINE_
#pragma warning(disable:4786)

#include <iostream>
#include "DataPreProcess.h"
#include "Configuration.h"
#include "FeatureExtractor.h"

using namespace std;

class SRLBaseline
{
public:
	//SRLBaseline() {}
	SRLBaseline(string configXml, string selectFeats);
	~SRLBaseline();	

public:

	//for now used
	void setDataPreProc(const DataPreProcess* dataPreProc);
	void SetPredicate(const vector<int>& vecPred);

protected:
	bool IsFilter(int nodeID, 
				  int intCurPd) const;

protected:
	const DataPreProcess	*m_dataPreProc;
	Configuration			m_configuration;
	FeatureExtractor		*m_featureExtractor;
	FeatureCollection		*m_featureCollection;
	vector<int>				m_featureNumbers;
    vector<string>			m_featurePrefixes;
	vector< vector<string> > m_selectFeatures;
	vector<int>    m_vecPredicate;
};

#endif
