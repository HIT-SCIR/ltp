#include "SRL_DLL.h"
#include "DepSRL.h"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

using namespace std;

static DepSRL g_depSRL;

// helper functions
int isLegalInput(const vector<string> &words, const vector<string> &POSs, const vector< pair<int, string> > &parse);


// Load Resources
int srl_load_resource(const string &modelFile)
{
    return g_depSRL.LoadResource(modelFile);
}

// Release Resources
int srl_release_resource()
{
    return g_depSRL.ReleaseResource();
}

/**
 *
 */
int srl_dosrl(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<pair<int, string> > &parse,
        vector<pair<int, vector<pair<string, pair<int, int> > > > > &vecSRLResult
) {
  vecSRLResult.clear();
  if (!isLegalInput(words, POSs, parse)) return -1;
  return g_depSRL.GetSRLResult(words, POSs, parse, vecSRLResult);
}


// helper functions


int isLegalInput(const vector<string> &words, const vector<string> &POSs, const vector< pair<int, string> > &parse)
{
  if (words.size() != POSs.size() || words.size() != parse.size()) {
    return false;
  }

  int len = words.size();
  for (int i = 0; i < len; ++ i) {
    if (words[i].empty() || POSs[i].empty()) {
      return false;
    }
    int father = parse[i].first;
    if (father < -1 || father >= len || parse[i].second.empty()) {
      return false;
    }
  }
  return true;
}
