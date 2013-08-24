/*
 * File Name     : SRLBaselineExt.h
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-8-21
 *
 */

#ifndef __SRL_BASELINE_EXT__
#define __SRL_BASELINE_EXT__

#include "SRLBaseline.h"
#include "Configuration.h"
#include "FeatureExtractor.h"

class SRLBaselineExt : public SRLBaseline
{
public:
    SRLBaselineExt(string configXml, string selectFeats);
    ~SRLBaselineExt();

public:
    void ExtractSrlFeatures(
        VecFeatForSent& vecAllFeatures,
        VecPosForSent& vecAllPos
        ) const;

    void ExtractPrgFeatures(
        vector< vector<string> >& vecPrgFeatures
        ) const;

    void convert2ConllFormat(
        vector<string>& vecRows
        ) const;

    void get_feature_config();

    void open_select_config(string selectConfig);

protected:
    bool IsFilter(int nodeID, int intCurPd) const;

};

#endif

