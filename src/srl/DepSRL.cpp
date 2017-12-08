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
#include "extractor/ExtractorFileToWordEmb.h"
#include "vector"
#include "dynet/dynet.h"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

using namespace std;

std::mutex DepSRL::mtx;

// Load necessary resources into memory
int DepSRL::LoadResource(const string &modelFile)
{
    dynet::DynetParams params;
    params.mem_descriptor = "2000";
    dynet::initialize(params);

    ifstream in(modelFile, ios::binary);
    if (!in) { return  -1;}
    boost::archive::binary_iarchive ia(in);
    ia >> piConfig;
    ia >> srlConfig;
    ia >> embedding;
    pi_model = new PiModel(piConfig);
    pi_model->loadDict(ia);
    pi_model->init();
    pi_model->loadModel(ia);
    pi_model->initEmbedding(embedding);
    srl_model = new SrlSrlModel(srlConfig);
    srl_model->loadDict(ia);
    srl_model->init();
    srl_model->loadModel(ia);
    srl_model->initEmbedding(embedding);
    m_resourceLoaded = true;
    return 0;
}

// Release all resources
int DepSRL::ReleaseResource()
{
    delete srl_model;
    delete pi_model;
    embedding.clear();
    m_resourceLoaded = false;
    return 0;
}

int DepSRL::GetSRLResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector< pair<int, string> > &parse,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    vecSRLResult.clear();
    SrlPiSample sentence;
    for (int j = 0; j < words.size(); ++j) {
        Word word(j, words[j], POSs[j], parse[j].first, parse[j].second, (j <= parse[j].first ? "before" : "after"), NIL_LABEL);
        sentence.push_back(word);
    }
    // pi prediction
    mtx.lock();
    {
        ComputationGraph hg;
        vector<Expression> adists = pi_model->label(hg, sentence);
        pi_model->ExtractResults(hg, adists, sentence);
    }
    mtx.unlock();
    if ( !sentence.getPredicateList().size() ) {
        // skip all processing if no predicate
        return 0;
    }
    // srl prediction
    mtx.lock();
    {
        ComputationGraph hg;
        vector<Expression> adists = srl_model->label(hg, sentence);
        srl_model->ExtractResults(hg, adists, sentence);
    }
    mtx.unlock();
    if (!FormResult(words, POSs, sentence.getPredicateList(), sentence, vecSRLResult))
      return -1;
    return 0;
}

int DepSRL::FormResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<int>    &VecAllPredicates,
        SrlPiSample& sentence,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
        )
{
    vecSRLResult.clear();
    vector< pair<int, int> > childArea;
    GetChildArea(sentence, childArea);
    for (size_t idx=0; idx<VecAllPredicates.size(); ++idx)
    {
        int predicate_position = VecAllPredicates[idx];
        vector< pair< string, pair< int, int > > > vecResultForOnePredicate;
        vector<string> args;
        for (int w = 0; w < sentence.size(); w++) args.push_back(sentence.getWord(w).getArgs()[idx]);
        ProcessOnePredicate(words, POSs, predicate_position, args, childArea, vecResultForOnePredicate);
        if (vecResultForOnePredicate.size())
          vecSRLResult.push_back(make_pair(predicate_position, vecResultForOnePredicate));
    }
    return 1;
}

void DepSRL::GetChildArea(SrlPiSample& sentence, vector< pair<int, int> >& childArea) {
  childArea.resize(sentence.size());
  for (int w = 0; w < sentence.size(); ++w) {
    childArea[w].first = childArea[w].second = w;
  }
  for (int w = 0; w < sentence.size(); ++w) {
    for (int p = sentence.getWord(w).getParent(); p != -1; p = sentence.getWord(p).getParent()) {
      if (w < childArea[p].first) childArea[p].first = w;
      if (w > childArea[p].second) childArea[p].second = w;
    }
  }
}

void DepSRL::ProcessOnePredicate(
        const vector<string>& vecWords,
        const vector<string>& vecPos,
        int intPredicates, 
        const vector<string>& args,
        const vector< pair<int, int> >& childArea,
        vector< pair< string, pair< int, int > > > &vecResultForOnePredicate
        )
{
  vecResultForOnePredicate.resize(0);

  //step1. insert label other than nil
  for (int j = 0; j < args.size(); ++j) {
    if (args[j] != NIL_LABEL) vecResultForOnePredicate.push_back(make_pair(args[j], childArea[j]));
  }

  //step2. process the collision
  ProcessCollisions(intPredicates, vecResultForOnePredicate);

  //step3. process the same tags
  // pass 当之前的arg概率小于0.5，而且该arg概率更大时，插入重复论元

  //step4. post process
   QTYArgsProcess(vecPos, vecResultForOnePredicate);
}

void DepSRL::ProcessCollisions(int intPredicates, vector<pair<string, pair<int, int> > > &results) {
  for (int j = 0; j < results.size(); ++j) {
    for (int k = 0; k < results.size(); ++k) {
      if ((results[k].second.first <= intPredicates && intPredicates <= results[k].second.second)
          ||
          (j != k && results[j].second.first <= results[k].second.first && results[k].second.second <= results[j].second.second)) {
        // k including predicate or j including k
        // remove k
        results.erase(results.begin() + k);
        ProcessCollisions(intPredicates, results);
        return;
      }
    }
  }
}

void DepSRL::QTYArgsProcess(const vector<string>& vecPos, vector< pair< string, pair< int, int > > > &vecResultForOnePredicate) const
{
  for (int j = 0; j < vecResultForOnePredicate.size(); ++j) {
    auto& res = vecResultForOnePredicate[j];
    if (res.first == S_QTY_ARG) {
      int k = res.second.first;
      for (; k <= res.second.second; ++k) {
        if (find(S_QTY_POS_PAT.begin(), S_QTY_POS_PAT.end(), vecPos[k]) != S_QTY_POS_PAT.end()) break;
      }
      if (k == res.second.second + 1) vecResultForOnePredicate.erase(vecResultForOnePredicate.begin() + j);
      QTYArgsProcess(vecPos, vecResultForOnePredicate);
      return;
    }
  }
}


void DepSRL::manageConfigPath(ModelConf &config, const string &dirPath) {
    config.model = dirPath + '/' + config.model;
}
