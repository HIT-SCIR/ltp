 #ifndef __RULENERECOG_H__
#define __RULENERECOG_H__

// #define STL_USING_ALL
// #include <STL.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "InitDic.h"

using namespace std;

struct RuleNode
{
	int nRuleBeg;    //rule begin position
	int nRuleEnd;    //rule end position
	int nRuleType;   //rule type
	int nRuleLen;    //rule length
	double nRulePro; //rule probability
};
typedef struct RuleNode RULENODE;

class Ruletest
{
public:
	Ruletest();
	void setObject(InitDic* dic);
	void RuleNErecog(vector<string>& vecNE,
		             vector< pair<string, string> >& vecpaSen);    
	 
private:
	void recogNtNr(vector< pair<string, string> >& vecpaSen,
		           unsigned int& pos, vector<string>& vecOut);

	void recogNm(vector< pair<string, string> >& vecpaSen,
		         unsigned int& pos, vector<string>& vecOut);
	
	void recogComplexNE(vector< pair<string, string> >& vecpaSen, 
		                unsigned int& pos, vector<string>& vecNE);
    void recogMissedNi(vector< pair<string, string> >& vecpaSen, 
		                unsigned int& pos, vector<string>& vecNE);
	void recogAbbreviation(vector< pair<string, string> >& vecpaSen, 
		                   unsigned int& pos, vector<string>& vecNE);
	void amendComplexNi(vector< pair<string, string> >& vecpaSen, 
		                unsigned int& pos, vector<string>& vecNE);
	
	void amendComplexNs(vector< pair<string, string> >& vecpaSen, 
		                unsigned int& pos, vector<string>& vecNE);
	void amendComplexNz(vector< pair<string, string> >& vecpaSen, 
						unsigned int& pos, vector<string>& vecNE);

	int match_SingleNE(vector<pair<string, string> >& vecpaSen,
		               unsigned int pos, string NEtype);

	int match_ComplexNE(vector<pair<string, string> >& vecpaSen,
		                unsigned int pos);

	int isRule(string& strPos, char* pRule, int& num);
	void correctNEResult(vector<string>& vecNE);
	void correctNsNiNztags(unsigned int begpos, unsigned int endpos,
						   int typeNum, vector<string>& vecNE);
	string getNEtype(int NEtypeNum);

	void addNETag(int begin, int len, string NEtype, vector<string>& vecOut);
	int findNEend(unsigned int begpos, vector<string>& vecNE);
	int findNsNiNzendWord(unsigned int begpos, vector< pair<string, string> >& vecpaSen, int typeNum);
	
private:
	InitDic* m_pdic;
	RULENODE Rulenode; //存放规则匹配的候选结果
	enum
	{
		PROBRULENum = 3570,
		NsendNum = 1,
        NiendNum = 2,
		NzendNum = 3,

		MaxRuleLen = 10,
		Maxchecklen = 5
	};	
};

#endif

