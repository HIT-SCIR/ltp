/*
 * File Name     : SRLBaseline.h
 * Author        : Frumes
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-8-21
 *
 */

#include "SRLBaseline.h"

SRLBaseline::SRLBaseline(string configXml, string selectFeats)
  : m_dataPreProc(NULL),
    m_featureExtractor(NULL),
    m_featureCollection(NULL)
{
}

SRLBaseline::~SRLBaseline()
{
  if (m_dataPreProc)       { delete m_dataPreProc; }
  if (m_featureCollection) { delete m_featureCollection; }
  if (m_featureExtractor)  { delete m_featureExtractor; }
}

// Check if the node will be filtered: only when the node 
// is predicate and punctation
inline bool SRLBaseline::IsFilter(int nodeID, int intCurPd) const
{
    DepNode depNode;
    m_dataPreProc->m_myTree->GetNodeValue(depNode, nodeID);

    //the punctuation nodes, current predicate node
    //changed for PTBtoDep, only filter the current predicate
    if(nodeID == intCurPd)
    {
        return 1;
    }
    else
    {
        return 0;
    }

    //return 0;
}


void SRLBaseline::SetPredicate(const vector<int>& vecPred)
{
    m_vecPredicate = vecPred;
}

void SRLBaseline::setDataPreProc(const DataPreProcess* dataPreProc)
{
    m_dataPreProc = dataPreProc;
}

bool SRLBaseline::isVerbPOS(const string& POS) const
{
    return m_configuration.is_verbPOS(POS);
}

