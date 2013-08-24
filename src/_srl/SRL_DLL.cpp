#include "SRL_DLL.h"
#include "DepSRL.h"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

using namespace std;

static DepSRL g_depSRL;
static vector< pair< int, vector< pair<string, pair< int, int > > > > > g_vecSRLResult;

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
        const vector< pair<int, string> > &parse)
{
    g_vecSRLResult.clear();
    if (0 == g_depSRL.GetSRLResult(words, POSs, NEs, parse, g_vecSRLResult)) return -1;;
    return g_vecSRLResult.size();
}

int GetSRLResult_size(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult)
{
    if (vecSRLResult.size() != g_vecSRLResult.size()) {
        cerr << "vecSRLResult size != g_vecSRLResult size" << endl;
        return -1;
    }
    int i = 0;
    for (; i < vecSRLResult.size(); ++i) {
        vecSRLResult[i].first = g_vecSRLResult[i].second.size();
    }
    return 0;
}

int GetSRLResult(
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult)
{
    if (vecSRLResult.size() != g_vecSRLResult.size()) {
        cerr << "vecSRLResult size != g_vecSRLResult size" << endl;
        return -1;
    }
    int i = 0;
    for (; i < vecSRLResult.size(); ++i) {
        if (vecSRLResult[i].second.size() != g_vecSRLResult[i].second.size()) {
            cerr << "vecSRLResult[i].second.size() != g_vecSRLResult[i].second.size()" << endl
                << "i = " << i << endl;
        }
        vecSRLResult[i].first = g_vecSRLResult[i].first;
        int j = 0;
        for (; j < g_vecSRLResult[i].second.size(); ++j) {
            vecSRLResult[i].second[j].first = g_vecSRLResult[i].second[j].first.c_str();
            vecSRLResult[i].second[j].second.first = g_vecSRLResult[i].second[j].second.first;
            vecSRLResult[i].second[j].second.second = g_vecSRLResult[i].second[j].second.second;
        }
    }
    return 0;
}

