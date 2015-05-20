/*
 * File name    : DepSRL.cpp
 * Author       : msmouse
 * Create Time  : 2009-09-19
 * Remark       : feature selection, post-process, result generation
 *
 * Updates by   : jiangfeng
 *
 */


#include "DepSRL.h"
#include "FeatureExtractor.h"
#include "Configuration.h"
#include "boost/bind.hpp"
#include "boost/algorithm/string.hpp"

// Load necessary resources into memory
int DepSRL::LoadResource(const string &ConfigDir)
{
    m_configXml = ConfigDir + "/Chinese.xml";
    m_selectFeats = ConfigDir + "/srl.cfg";
    // load srl and prg model
    m_srlModel = new maxent::ME_Model;
    bool tag = m_srlModel->load(ConfigDir + "/srl.model");
    if(!tag) {
      return 0;
    }

    m_prgModel = new maxent::ME_Model;
    tag = m_prgModel->load(ConfigDir + "/prg.model");
    if(!tag) {
      return 0;
    }

    m_resourceLoaded = true;

    return true;
}

// Release all resources
int DepSRL::ReleaseResource()
{
    delete m_srlModel;
    delete m_prgModel;

    m_resourceLoaded = false;

    return 1;
}
string DepSRL::GetConfigXml()
{
    return m_configXml;
}
string DepSRL::GetSelectFeats()
{
    return m_selectFeats;
}
int DepSRL::GetSRLResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    LTPData ltpData;
    ltpData.vecWord = words;
    ltpData.vecPos  = POSs;
    ltpData.vecNe   = NEs;

    GetParAndRel(parse, ltpData.vecParent, ltpData.vecRelation);

    // construct a DataPreProcess instance
    DataPreProcess* dataPreProc = new DataPreProcess(&ltpData);

    SRLBaselineExt * m_srlBaseline=new SRLBaselineExt(GetConfigXml(),GetSelectFeats());
    // extract features !
    m_srlBaseline->setDataPreProc(dataPreProc);

    // GetPredicateFromSentence(POSs,predicates);
    vector<int> predicates;
    GetPredicateFromSentence(predicates,m_srlBaseline);

    // return GetSRLResult(words, POSs, NEs, parse, predicates, vecSRLResult);
    return GetSRLResult(ltpData, predicates, vecSRLResult,m_srlBaseline);
}

// produce DepSRL result for a sentence
/*
int DepSRL::GetSRLResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        const vector<int> &predicates,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    LTPData ltpData;
    ltpData.vecWord = words;
    ltpData.vecPos  = POSs;
    ltpData.vecNe   = NEs;

    // transform LTP parse result to parent-relation format
    GetParAndRel(parse, ltpData.vecParent, ltpData.vecRelation);

    return GetSRLResult(ltpData, predicates, vecSRLResult);
}
*/

// produce DepSRL result for a sentence
int DepSRL::GetSRLResult(
        const LTPData     &ltpData,
        const vector<int> &predicates,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult,
        SRLBaselineExt * m_srlBaseline) {
    vecSRLResult.clear();

    if ( !m_resourceLoaded ) {
        cerr<<"Resources not loaded."<<endl;
        return 0;
    }

    if ( !predicates.size() ) {
        // skip all processing if no predicate
        return 1;
    }

    VecFeatForSent vecAllFeatures;   //the features interface for SRLBaseline
    VecPosForSent  vecAllPos;        //the constituent position vector
    vector< vector< pair<string, double> > > vecAllPairMaxArgs;
    vector< vector< pair<string, double> > > vecAllPairNextArgs;

    // extract features
    if (!ExtractSrlFeatures(ltpData, predicates,vecAllFeatures,vecAllPos,m_srlBaseline))
        return 0;

    // predict
    if (!Predict(vecAllFeatures,vecAllPairMaxArgs,vecAllPairNextArgs))
        return 0;

    // form the result
    if (!FormResult(
                ltpData.vecWord,ltpData.vecPos, predicates,vecAllPos,
                vecAllPairMaxArgs,vecAllPairNextArgs,
                vecSRLResult
                )
       ) return 0;

    // rename arguments to short forms (ARGXYZ->AXYZ)
    if (!RenameArguments(vecSRLResult)) return 0;
    delete m_srlBaseline;

    return 1;
}

int DepSRL::ExtractSrlFeatures(
        const LTPData     &ltpData,
        const vector<int> &VecAllPredicates,
        VecFeatForSent    &vecAllFeatures,
        VecPosForSent     &vecAllPos,
        SRLBaselineExt* m_srlBaseline
        )
{
    vecAllFeatures.clear();
    vecAllPos.clear();

    /*
    // construct a DataPreProcess instance
    DataPreProcess* dataPreProc = new DataPreProcess(&ltpData);

    // extract features !
    m_srlBaseline->setDataPreProc(dataPreProc);
    */

    m_srlBaseline->SetPredicate(VecAllPredicates);
    m_srlBaseline->ExtractSrlFeatures(vecAllFeatures, vecAllPos);

    return 1;
}

int DepSRL::Predict(
        VecFeatForSent &vecAllFeatures,
        vector< vector< pair<string, double> > > &vecAllPairMaxArgs,
        vector< vector< pair<string, double> > > &vecAllPairNextArgs
        )
{
    vector< pair<string, double> > vecPredPairMaxArgs;
    vector< pair<string, double> > vecPredPairNextArgs;

    for(VecFeatForSent::iterator predicate_iter = vecAllFeatures.begin();
            predicate_iter != vecAllFeatures.end();
            ++predicate_iter
       ){// for each predicate
        vecPredPairMaxArgs.clear();
        vecPredPairNextArgs.clear();

        for(VecFeatForVerb::iterator position_iter = (*predicate_iter).begin();
                position_iter != (*predicate_iter).end();
                ++position_iter
           ) {// for each position
            vector<pair<string,double> > outcome;

            maxent::ME_Sample sample(*position_iter);
            m_srlModel->predict(sample, outcome);
            // m_srlModel->eval_all((*position_iter),outcome);

            vecPredPairMaxArgs.push_back(outcome[0]);
            vecPredPairNextArgs.push_back(outcome[1]);
        }

        vecAllPairMaxArgs.push_back(vecPredPairMaxArgs);
        vecAllPairNextArgs.push_back(vecPredPairNextArgs);
    }

    return 1;
}

int DepSRL::FormResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<int>          &VecAllPredicates,
        VecPosForSent        &vecAllPos,
        vector< vector< pair<string, double> > > &vecAllPairMaxArgs,
        vector< vector< pair<string, double> > > &vecAllPairNextArgs,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    vecSRLResult.clear();
    vector< pair< string, pair< int, int > > > vecResultForOnePredicate;

    for (size_t idx=0; idx<VecAllPredicates.size(); ++idx)
    {
        int predicate_position = VecAllPredicates[idx];

        vecResultForOnePredicate.clear();

        ProcessOnePredicate(
                words, POSs, predicate_position, vecAllPos[idx], 
                vecAllPairMaxArgs[idx], vecAllPairNextArgs[idx], 
                vecResultForOnePredicate
                );

        if ( vecResultForOnePredicate.size() > 1 ) {
            vecResultForOnePredicate.pop_back(); // pop the "V" arg
            vecSRLResult.push_back(make_pair(predicate_position,vecResultForOnePredicate));
        }
        //vecResultForOnePredicate.pop_back(); // pop the "V" arg
        //vecSRLResult.push_back(make_pair(predicate_position,vecResultForOnePredicate));
    }

    return 1;
}

// result forming form one predicate, based on hjliu's original function
void DepSRL::ProcessOnePredicate(
        const vector<string>& vecWords,
        const vector<string>& vecPos,
        int intPredicates, 
        const vector< pair<int, int> >& vecPairPS,
        const vector< pair<string, double> >& vecPairMaxArgs, 
        const vector< pair<string, double> >& vecPairNextArgs,
        vector< pair< string, pair< int, int > > > &vecResultForOnePredicate
        )
{
    vector< pair<int, int> > vecPairPSBuf;
    vector< pair<string, double> > vecPairMaxArgBuf;
    vector< pair<string, double> > vecPairNextArgBuf;

    //step1. remove the null label
    vector< pair<string, double> > vecPairNNLMax;
    vector< pair<string, double> > vecPairNNLNext; 
    vector< pair<int, int> > vecPairNNLPS;
    RemoveNullLabel(vecPairMaxArgs, vecPairNextArgs, vecPairPS, vecPairNNLMax, vecPairNNLNext, vecPairNNLPS);

    // step 2. insert the args
    vector<int> vecItem;
    for (int index = 0; index < vecPairNNLPS.size(); index++)
    {
        InsertOneArg( vecPairNNLPS.at(index), vecPairNNLMax.at(index), vecPairNNLNext.at(index), vecPairPSBuf, vecPairMaxArgBuf, vecPairNextArgBuf ) ;
    }

    // step 3. insert predicate node
    if ( IsInsertPredicate(intPredicates, vecPairMaxArgBuf, vecPairPSBuf) )
    {
        pair<int, int> prPdPS;
        pair<string, double> prPdArg;
        prPdPS.first   = intPredicates;
        prPdPS.second  = intPredicates;
        prPdArg.first  = S_PD_ARG;
        prPdArg.second = 1;

        vecPairPSBuf.push_back(prPdPS);
        vecPairMaxArgBuf.push_back(prPdArg);
        vecPairNextArgBuf.push_back(prPdArg);
    }

    // step 4. post process
    PostProcess(vecPos, vecPairPS, vecPairMaxArgs, vecPairNextArgs, vecPairPSBuf, vecPairMaxArgBuf, vecPairNextArgBuf);

    // put into output vector
    for (int index = 0; index < vecPairPSBuf.size(); index++)
    {
        vecResultForOnePredicate.push_back(make_pair(vecPairMaxArgBuf[index].first, vecPairPSBuf[index]));
    }
}

void DepSRL::RemoveNullLabel(const vector< pair<string, double> >& vecPairMaxArgs, 
        const vector< pair<string, double> >& vecPairNextArgs, 
        const vector< pair<int, int> >& vecPairPS, 
        vector< pair<string, double> >& vecPairNNLMax, 
        vector< pair<string, double> >& vecPairNNLNext, 
        vector< pair<int, int> >& vecPairNNLPS) const
{
    vecPairNNLMax.clear();
    vecPairNNLNext.clear();
    vecPairNNLPS.clear();
    for (int index = 0; index < vecPairMaxArgs.size(); index++)
    {
        if ( vecPairMaxArgs.at(index).first.compare(S_NULL_ARG) )
        {
            vecPairNNLMax.push_back(vecPairMaxArgs.at(index));
            vecPairNNLNext.push_back(vecPairNextArgs.at(index));
            vecPairNNLPS.push_back(vecPairPS.at(index));
        }
    }
}

void DepSRL::InsertOneArg(const pair<int, int>& pArgPS, 
        const pair<string, double>& pMaxArg, 
        const pair<string, double>& pNextArg,                                    
        vector< pair<int, int> >& vecPairPSBuf, 
        vector< pair<string, double> >& vecPairMaxArgBuf, 
        vector< pair<string, double> >& vecPairNextArgBuf) const
{
    // 2.1. process the collision
    vector<int> vctCol;
    FindCollisionCand(vecPairPSBuf, pArgPS, vctCol);
    if ( !IsInsertNColLabel(vctCol, pMaxArg, vecPairMaxArgBuf, vecPairNextArgBuf, vecPairPSBuf) ) 
    {
        // insert current node
        // vecPairMaxArgBuf.push_back(pMaxArg);
        // vecPairNextArgBuf.push_back(pNextArg);
        // vecPairPSBuf.push_back(pArgPS);

        // process next arg
        return;
    }

    // 2.2. process the same args
    vector<int> vctSame;
    vector<int> vctSameDel;
    FindSameLabelCand(vecPairMaxArgBuf, pMaxArg, vctSame);
    if ( !IsInsertSameLabel(vctSame, pMaxArg, vecPairMaxArgBuf, vecPairNextArgBuf, vecPairPSBuf, vctSameDel) )
    {
        // insert current node
        // vecPairMaxArgBuf.push_back(pMaxArg);
        // vecPairNextArgBuf.push_back(pNextArg);
        // vecPairPSBuf.push_back(pArgPS);

        // process next arg
        return; 
    }

    // 2.3 insert current node
    // remove collisions and same-args
    // BOOST_FOREACH (int id, vctCol) {
    for(int id = 0; id < vctCol.size(); id++) {
        vecPairMaxArgBuf[id].second  = -1;
        vecPairNextArgBuf[id].second = -1;
        vecPairPSBuf[id].second      = -1;
    }
    // BOOST_FOREACH (int id, vctSameDel) {
    for(int id = 0; id < vctSameDel.size(); id++) {
        vecPairMaxArgBuf[id].second  = -1;
        vecPairNextArgBuf[id].second = -1;
        vecPairPSBuf[id].second      = -1;
    }
    vecPairMaxArgBuf.erase(
            remove_if(
                vecPairMaxArgBuf.begin(),
                vecPairMaxArgBuf.end(),
                boost::bind(
                    less<double>(),
                    boost::bind(
                        &pair<string,double>::second,
                        _1
                        ),
                    0
                    )
                ),
            vecPairMaxArgBuf.end()
            );
    vecPairNextArgBuf.erase(
            remove_if(
                vecPairNextArgBuf.begin(),
                vecPairNextArgBuf.end(),
                boost::bind(
                    less<double>(),
                    boost::bind(
                        &pair<string,double>::second,
                        _1
                        ),
                    0
                    )
                ),
            vecPairNextArgBuf.end()
            );
    vecPairPSBuf.erase(
            remove_if(
                vecPairPSBuf.begin(),
                vecPairPSBuf.end(),
                boost::bind(
                    less<int>(),
                    boost::bind(
                        &pair<int,int>::second,
                        _1
                        ),
                    0
                    )
                ),
            vecPairPSBuf.end()
            );
    vecPairMaxArgBuf.push_back(pMaxArg);
    vecPairNextArgBuf.push_back(pNextArg);
    vecPairPSBuf.push_back(pArgPS);
}

bool DepSRL::IsInsertPredicate(int intPredicate, 
        vector< pair<string, double> >& vecPairMaxArgBuf, 
        vector< pair<int, int> >& vecPairPSBuf) const
{
    for(int index = 0; index < vecPairPSBuf.size(); index++)
    {
        if ( (vecPairPSBuf.at(index).first <= intPredicate) &&
                (vecPairPSBuf.at(index).second >= intPredicate) )
        {
            vecPairPSBuf.at(index).first = intPredicate;
            vecPairPSBuf.at(index).second = intPredicate;
            vecPairMaxArgBuf.at(index).first = S_PD_ARG;
            vecPairMaxArgBuf.at(index).second = 1;

            return 0;
        }
    }

    return 1;
}

/*
void DepSRL::TransVector(const vector<const char*>& vecInStr, 
                               vector<string>& vecOutStr) const
{
    vector<const char*>::const_iterator itInStr;
    itInStr = vecInStr.begin();
    while (itInStr != vecInStr.end()) 
    {
        vecOutStr.push_back(*itInStr);
        itInStr++;
    }
}
*/

void DepSRL::GetParAndRel(const vector< pair<int, string> >& vecParser, 
        vector<int>& vecParent, 
        vector<string>& vecRelation) const
{
    vector< pair<int, string> >::const_iterator itParser;
    pair<int, string> pairParser;

    itParser = vecParser.begin();
    while(itParser != vecParser.end())
    {
        pairParser = *itParser;
        vecParent.push_back(pairParser.first);
        vecRelation.push_back(pairParser.second);
        ++ itParser;
    }
}

void DepSRL::GetPredicateFromSentence(const vector<string>& vecPos, 
        vector<int>& vecPredicate,SRLBaselineExt* m_srlBaseline) const
{
    int index;
    vector<string>::const_iterator itPos;
    index = 0;
    itPos = vecPos.begin();
    while (itPos != vecPos.end())
    {
        if (m_srlBaseline->isVerbPOS(*itPos))
        {
            vecPredicate.push_back(index);
        }

        ++ index;
        ++ itPos;
    }
}

void DepSRL::GetPredicateFromSentence(vector<int>& vecPredicate,SRLBaselineExt * m_srlBaseline) const
{
    /* extract features for each word in sentence */
    vector< vector<string> > vecFeatures;
    m_srlBaseline->ExtractPrgFeatures(vecFeatures);

    /* predict */
    for (size_t i = 0; i < vecFeatures.size(); ++i)
    {
        maxent::ME_Sample sample(vecFeatures[i]);
        vector< pair<string, double> > prediction;
        m_prgModel->predict(sample, prediction);
        if (prediction[0].first == "Y")
            vecPredicate.push_back(i);
    }
}

void DepSRL::PostProcess(const vector<string>& vecPos,
        const vector< pair<int, int> >& vecPairPS,
        const vector< pair<string, double> >& vecPairMaxArgs,
        const vector< pair<string, double> >& vecPairNextArgs,
        vector< pair<int, int> >& vecPairPSBuf,
        vector< pair<string, double> >& vecPairMaxArgsBuf,
        vector< pair<string, double> >& vecPairNextArgsBuf) const
{
    // step 1. process QTY args
    QTYArgsProcess(vecPos, vecPairPSBuf, vecPairMaxArgsBuf, vecPairNextArgsBuf);

    // step 2. process PSR-PSE arg
    PSERArgsProcess(S_ARG0_TYPE, vecPos, vecPairPS, vecPairMaxArgs, vecPairNextArgs, vecPairPSBuf, vecPairMaxArgsBuf, vecPairNextArgsBuf);
}


void DepSRL::FindCollisionCand(const vector< pair<int, int> >& vecPairPSCands, 
        const pair<int, int>& pairCurPSCand, 
        vector<int>& vecPairColPSCands) const
{
    vecPairColPSCands.clear();
    for (int index = 0; index < vecPairPSCands.size(); index++)
    {
        if ( ((pairCurPSCand.first >= vecPairPSCands.at(index).first) && (pairCurPSCand.first <= vecPairPSCands.at(index).second)) ||
                ((pairCurPSCand.second >= vecPairPSCands.at(index).first) && (pairCurPSCand.second <= vecPairPSCands.at(index).second)) || 
                ((pairCurPSCand.first <= vecPairPSCands.at(index).first) && (pairCurPSCand.second >= vecPairPSCands.at(index).second)) )
        {
            vecPairColPSCands.push_back(index);
        }
    }
}

// format: (argType, argProp)
void DepSRL::FindSameLabelCand(
        const vector< pair<string, double> >& vecPairArgCands,
        const pair<string, double>& pairCurArgCand,
        vector<int>& vecPairSameArgCands) const
{
    vecPairSameArgCands.clear();
    for (int index = 0; index < vecPairArgCands.size(); index++)
    {
        if ( !pairCurArgCand.first.compare(vecPairArgCands.at(index).first) )
        {
            vecPairSameArgCands.push_back(index);
        }
    }
}

void DepSRL::QTYArgsProcess(
        const vector<string>& vecPos,
        vector< pair<int, int> >& vecPairPSBuf,
        vector< pair<string, double> >& vecPairMaxArgsBuf,
        vector< pair<string, double> >& vecPairNextArgsBuf) const
{
    vector< pair<int, int> > vecPairPSTemp(vecPairPSBuf);
    vector< pair<string, double> > vecPairMaxArgsTemp(vecPairMaxArgsBuf);
    vector< pair<string, double> > vecPairNextArgsTemp(vecPairNextArgsBuf);

    vecPairPSBuf.clear();
    vecPairMaxArgsBuf.clear();
    vecPairNextArgsBuf.clear();
    // process rule : if (arg_type is "*-QTY") then the pos_pattern must (AD|CD|M)+
    // else must process: if next arg_type is "NULL" then drop this candidate
    // else replace with the next arg_type
    for (int index = 0; index < vecPairPSTemp.size(); index++)
    {
        if ( (vecPairMaxArgsTemp.at(index).first.find(S_QTY_ARG) != string::npos) &&
                !IsPosPattern(vecPairPSTemp.at(index).first, vecPairPSTemp.at(index).second, vecPos, S_QTY_POS_PAT) )
        {
            if ( !vecPairNextArgsTemp.at(index).first.compare(S_NULL_ARG) )
            {
                continue;
            }
            else
            {
                vecPairMaxArgsTemp.at(index) = vecPairNextArgsTemp.at(index);
            }
        }

        // add to candidate
        vecPairPSBuf.push_back(vecPairPSTemp.at(index));
        vecPairMaxArgsBuf.push_back(vecPairMaxArgsTemp.at(index));
        vecPairNextArgsBuf.push_back(vecPairNextArgsTemp.at(index));
    }
}

void DepSRL::PSERArgsProcess(
        const string& strArgPrefix,
        const vector<string>& vecPos, 
        const vector< pair<int, int> >& vecPairPS,
        const vector< pair<string, double> >& vecPairMaxArgs,
        const vector< pair<string, double> >& vecPairNextArgs,
        vector< pair<int, int> >& vecPairPSBuf, 
        vector< pair<string, double> >& vecPairMaxArgsBuf, 
        vector< pair<string, double> >& vecPairNextArgsBuf) const
{
    vector<int> vecPSRIndex;
    vector<int> vecPSEIndex;
    pair<int, int> pArgPS;
    pair<string, double> pMaxArg;
    pair<string, double> pNextArg;

    string psrArgType = strArgPrefix + S_HYPHEN_TAG + S_PSR_ARG;
    string pseArgType = strArgPrefix + S_HYPHEN_TAG + S_PSE_ARG;
    // step 1. find the PSR and PSE args index
    for (int index = 0; index < vecPairPSBuf.size(); index++)
    {
        if (vecPairMaxArgsBuf.at(index).first.find(psrArgType) != string::npos)
        {
            vecPSRIndex.push_back(index);
        }

        if (vecPairMaxArgsBuf.at(index).first.find(pseArgType) != string::npos)
        {
            vecPSEIndex.push_back(index);
        }
    }

    // step 2. check if matched
    if ( vecPSRIndex.empty() &&
            !vecPSEIndex.empty() )
    {
        // process the PSE args
        if ( IsMaxPropGreaterThreshold(I_ARG_THRESHOLD_VAL, vecPSEIndex, vecPairMaxArgsBuf) &&
                FindArgFromDropCand(psrArgType, vecPairPS, vecPairMaxArgs, vecPairNextArgs, pArgPS, pMaxArg, pNextArg) )
        {
            //find the matched arg-type
            InsertOneArg( pArgPS, pMaxArg, pNextArg, vecPairPSBuf, vecPairMaxArgsBuf, vecPairNextArgsBuf );
        }
    }
    else if ( !vecPSRIndex.empty() &&
            vecPSEIndex.empty() )
    {
        // process the PSR args
        // process the PSE args
        if ( IsMaxPropGreaterThreshold(I_ARG_THRESHOLD_VAL, vecPSRIndex, vecPairMaxArgsBuf) &&
                FindArgFromDropCand(pseArgType, vecPairPS, vecPairMaxArgs, vecPairNextArgs, pArgPS, pMaxArg, pNextArg) )
        {
            //find the matched arg-type
            InsertOneArg( pArgPS, pMaxArg, pNextArg, vecPairPSBuf, vecPairMaxArgsBuf, vecPairNextArgsBuf );
        }
    }

}

bool DepSRL::FindArgFromDropCand(
        const string& strArgPat,
        const vector< pair<int, int> >& vecPairPS,
        const vector< pair<string, double> >& vecPairMaxArgs,
        const vector< pair<string, double> >& vecPairNextArgs,
        pair<int, int>& pArgPS,
        pair<string, double>& pMaxArg,
        pair<string, double>& pNextArg) const
{
    int maxIndex = -1;
    int flag = -1;
    double maxProp = 0;

    for (int index = 0; index < vecPairPS.size(); index++)
    {
        if ( (vecPairMaxArgs.at(index).first.find(strArgPat) != string::npos) &&
                (vecPairMaxArgs.at(index).second > maxProp) )
        {
            maxIndex = index;
            maxProp = vecPairMaxArgs.at(index).second;
            flag = 1;
        }
        else if ( (vecPairNextArgs.at(index).first.find(strArgPat) != string::npos) &&
                (vecPairNextArgs.at(index).second > maxProp) )
        {
            maxIndex = index;
            maxProp = vecPairNextArgs.at(index).second;
            flag = 0;
        }
    }

    if ( (flag == -1) || (maxProp < 0.01) )
    {
        return 0;
    }
    else if (flag == 1)
    {
        pMaxArg = vecPairMaxArgs.at(maxIndex);
    }
    else
    {
        pMaxArg = vecPairNextArgs.at(maxIndex);
    }

    pArgPS = vecPairPS.at(maxIndex);
    pNextArg = vecPairNextArgs.at(maxIndex);
    return 1;
}

void DepSRL::ReplaceArgFromNextProp(
        const vector<int>& vecIndex,
        vector< pair<int, int> >& vecPairPSBuf, 
        vector< pair<string, double> >& vecPairMaxArgsBuf, 
        vector< pair<string, double> >& vecPairNextArgsBuf) const
{
    int delIndex = 0;
    // if next arg_type is "NULL" then drop this candidate
    // else replace with the next arg_type
    for (int index = 0; index < vecIndex.size(); index++)
    {
        if ( !vecPairNextArgsBuf.at(vecIndex.at(index)).first.compare(S_NULL_ARG) )
        {
            vecPairPSBuf.erase( vecPairPSBuf.begin() + vecIndex.at(index) - delIndex );
            vecPairMaxArgsBuf.erase( vecPairMaxArgsBuf.begin() + vecIndex.at(index) - delIndex );
            vecPairNextArgsBuf.erase( vecPairNextArgsBuf.begin() + vecIndex.at(index) - delIndex );

            delIndex++;
        }
        else
        {
            vecPairMaxArgsBuf.at(vecIndex.at(index)) = vecPairNextArgsBuf.at(vecIndex.at(index));
        }
    }
}

bool DepSRL::IsPosPattern(
        int intBegin,
        int intEnd,
        const vector<string>& vecPos,
        const string& strPattern) const
{
    vector<string> vecItem;
    boost::algorithm::split(vecItem, strPattern, boost::is_any_of("|"));

    for (int index = intBegin; index < intEnd; index++)
    {
        if ( find(vecItem.begin(), vecItem.end(), vecPos.at(index)) == vecItem.end() )
        {
            return 0;
        }
    }

    return 1;
}

bool DepSRL::IsMaxPropGreaterThreshold(
        double dThreSholdVal,
        const vector<int>& vecIndex,
        const vector< pair<string, double> >& vecPairMaxArgsBuf) const
{
    vector<int>::const_iterator itIndex;

    itIndex = vecIndex.begin();
    while (itIndex != vecIndex.end())
    {
        if (vecPairMaxArgsBuf.at(*itIndex).second >= dThreSholdVal)
        {
            return 1;
        }

        ++ itIndex;
    }

    return 0;
}


bool DepSRL::IsInsertNColLabel(
        const vector<int>& vecCol,
        const pair<string, double>& pArgCand,
        vector< pair<string, double> >& vecPairMaxArgBuf,
        vector< pair<string, double> >& vecPairNextArgBuf,
        vector< pair<int, int> >& vecPairPSBuf) const
{
    int id;
    int isPSColInsert = 1;
    if ( !vecCol.empty() )
    {
        for (id = 0; id < vecCol.size(); id++)
        {
            // P(Ci) > P(A), no insert
            if ( vecPairMaxArgBuf.at(vecCol.at(id)).second > pArgCand.second)
            {
                // isPSColInsert = 0;
                // break;
                return 0;
            }
        }

        /*
        // delete the collision nodes
        if (isPSColInsert)
        {
            for (id = 0; id < vecCol.size(); id++)
            {
                vecPairMaxArgBuf.erase(vecPairMaxArgBuf.begin() + vecCol.at(id) - id);
                vecPairNextArgBuf.erase(vecPairNextArgBuf.begin() + vecCol.at(id) - id );
                vecPairPSBuf.erase(vecPairPSBuf.begin() + vecCol.at(id) - id);
            }

            return 1;
        }

        return 0;
        */

    }

    return 1;
}

bool DepSRL::IsInsertSameLabel(
        const vector<int>& vecSame, 
        const pair<string, double>& pArgCand,
        vector< pair<string, double> >& vecPairMaxArgBuf, 
        vector< pair<string, double> >& vecPairNextArgBuf,
        vector< pair<int, int> >& vecPairPSBuf,
        vector<int>& vecSameDel) const
{
    int id;
    int isArgSameInsert = 1;

    // P(A) <  0.4
    if (pArgCand.second < 0.4)
    {
        isArgSameInsert = 0;
    }

    if ( !vecSame.empty() )
    {
        for (id = 0; id < vecSame.size(); id++)
        {
            // P(Ei) < P(A) < 0.5, insert
            if ( (vecPairMaxArgBuf.at(vecSame.at(id)).second < 0.5) &&
                    (vecPairMaxArgBuf.at(vecSame.at(id)).second < pArgCand.second) )
            {
                vecSameDel.push_back(vecSame.at(id));
                isArgSameInsert = 1;
            }
        }

        //delete the  small prob nodes
        if (isArgSameInsert)
        {
            // for (id = 0; id < vecArgDel.size(); id++)
            // {
            //     vecPairMaxArgBuf.erase(vecPairMaxArgBuf.begin() + vecArgDel.at(id) - id);
            //     vecPairNextArgBuf.erase(vecPairNextArgBuf.begin() + vecArgDel.at(id) - id);
            //     vecPairPSBuf.erase(vecPairPSBuf.begin() + vecArgDel.at(id) - id);
            // }

            return 1;
        }

        return 0;
    }
    else
    {
        return 1;
    }

}

int DepSRL::RenameArguments(
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    for (vector< pair< int, vector< pair< string, pair< int, int > > > > >::iterator 
            predicate_iter = vecSRLResult.begin();
            predicate_iter != vecSRLResult.end();
            ++predicate_iter
        )
    {
        for(vector< pair< string, pair< int, int > > >::iterator
                argument_iter = predicate_iter->second.begin();
                argument_iter != predicate_iter->second.end();
                ++argument_iter
           )
        {
            if (argument_iter->first.substr(0,3) == "ARG") {
                argument_iter->first = "A" + argument_iter->first.substr(3);
            }
        }
    }

    return 1;
}
