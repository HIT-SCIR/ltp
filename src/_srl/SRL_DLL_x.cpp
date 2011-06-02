#include "SRL_DLL.h"

int SRL(
		const vector<string> &words,
		const vector<string> &POSs,
		const vector<string> &NEs,
		const vector< pair<int, string> > &parse,
		vector< pair< int, vector< pair<const char *, pair< int, int > > > > > &vecSRLResult
		)
{
	vecSRLResult.clear();
	int resultNum = DoSRL(words, POSs, NEs, parse);
	if (resultNum < 0) return -1;
	if (resultNum == 0) return 0;
	vecSRLResult.resize(resultNum);
	if (0 != GetSRLResult_size(vecSRLResult)) return -1;
	int i = 0;
	for (; i < resultNum; ++i) {
		vecSRLResult[i].second.resize( vecSRLResult[i].first );
	}
	return GetSRLResult(vecSRLResult);
}
