/*
 * File Name     : SRLBaseline.h
 * Author        : Frumes
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-8-21
 *
 */

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
        SRLBaseline(string configXml, string selectFeats);
        ~SRLBaseline();

    public:
        void setDataPreProc(const DataPreProcess* dataPreProc);
        void SetPredicate(const vector<int>& vecPred);
        bool isVerbPOS(const string& POS) const;

    protected:
        bool IsFilter(int nodeID, int intCurPd) const;

    protected:
        const DataPreProcess     *m_dataPreProc;
        Configuration            m_configuration;
        FeatureExtractor         *m_featureExtractor;
        FeatureCollection        *m_featureCollection;
        vector<int>              m_prgFeatureNumbers;
        vector<int>              m_srlFeatureNumbers;
        vector<string>           m_prgFeaturePrefixes;
        vector<string>           m_srlFeaturePrefixes;
        vector<int>              m_vecPredicate;
        vector< vector<string> > m_srlSelectFeatures;
};

#endif

