#include "SRL_DLL.h"
#include "DepSRL.h"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

using namespace std;

static DepSRL g_depSRL;

// Load Resources
int SRL_LoadResource(const string &ConfigDir)
{
    if (0 == g_depSRL.LoadResource(ConfigDir)) return -1;
    return 0;
}

// Release Resources
int SRL_ReleaseResource()
{
    if (0 == g_depSRL.ReleaseResource()) return -1;
    return 0;
}

// perform SRL
int DoSRL(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult)
{
  if (words.size() != POSs.size()
      || words.size() != parse.size()
      || words.size() != NEs.size()) {
    return -1;
  }

  int len = words.size();
  for (int i = 0; i < len; ++ i) {
    if (words[i].empty() || POSs[i].empty() || NEs.empty()) {
      return -1;
    }
    int father = parse[i].first;
    if (father < -1 || father >= len || parse[i].second.empty()) {
      return -1;
    }
  }

  tmp_vecSRLResult.clear();

  if (0 == g_depSRL.GetSRLResult(words, POSs, NEs, parse, tmp_vecSRLResult)) {
    return -1;
  }

  return tmp_vecSRLResult.size();
}

int GetSRLResult_size(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult)
{
    if (vecSRLResult.size() != tmp_vecSRLResult.size()) {
        cerr << "vecSRLResult size != tmp_vecSRLResult size" << endl;
        return -1;
    }
    int i = 0;
    for (; i < vecSRLResult.size(); ++i) {
        vecSRLResult[i].first = tmp_vecSRLResult[i].second.size();
    }
    return 0;
}

int GetSRLResult(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &tmp_vecSRLResult)
{
    if (vecSRLResult.size() != tmp_vecSRLResult.size()) {
        cerr << "vecSRLResult size != tmp_vecSRLResult size" << endl;
        return -1;
    }
    int i = 0;
    for (; i < vecSRLResult.size(); ++i) {
        if (vecSRLResult[i].second.size() != tmp_vecSRLResult[i].second.size()) {
            cerr << "vecSRLResult[i].second.size() != tmp_vecSRLResult[i].second.size()" << endl
                << "i = " << i << endl;
        }
        vecSRLResult[i].first = tmp_vecSRLResult[i].first;
        int j = 0;
        for (; j < tmp_vecSRLResult[i].second.size(); ++j) {
            vecSRLResult[i].second[j].first = tmp_vecSRLResult[i].second[j].first.c_str();
            vecSRLResult[i].second[j].second.first = tmp_vecSRLResult[i].second[j].second.first;
            vecSRLResult[i].second[j].second.second = tmp_vecSRLResult[i].second[j].second.second;
        }
    }
    return 0;
}

