       #ifndef __INITDIC_H__
#define __INITDIC_H__

// #define STL_USING_ALL
// #include <STL.h>
#include <map>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <set>

using namespace std;

///////////////////////////////////////////////////////////////
//	结构体名 : probnode
//	描    述 : 用于存放词性串规则
//	功    能 : 
//	历史记录 : 
//	使用说明 : 
//	作    者 : taozi
//	时    间 : 2006年6月5日
//	备    注 : 
///////////////////////////////////////////////////////////////
struct probnode
{
	char* probrule;    //词性串
	double probability;//词性串对应的概率
	int NEtypeNum;     //词性串对应的NE类别号
};
typedef struct probnode PROBNODE;

struct indexnode
{
	int begIndex;
	int endIndex;
};
typedef struct indexnode RULEIDXNODE;

class InitDic
{
public:
	InitDic();
	~InitDic();
	int loadObver(char* infile="DATA");
	int loadState(char* infile="DATA");
	int loadRule(char* infile="DATA");
	void releaseRes();
	string getStatestr(int stateIndex);
	int getObserIndex(string& obser);
	int getStateIndex(string& state);
	int getWordIndexInMap(map<string, int>& mapName, string& word);
	void getRuleIndex(string& strRule, int& begIdx, int& endIdx);

private:
	int loadProrule(char* infile);
	void setRuleindex(string strPosRule, int index);
    
	int addRuletoMap(char* infile, map<string, int>& mapName, int index); 
	int addRuletoSet(char* infile, set<string>& setName);

public:
	map<string, int> m_mapObserstr2int;
//	map<int, string> m_mapObserint2str;
	map<string, int> m_mapStatestr2int;
	map<int, string> m_mapStateint2str;

	map<string, int> m_mapNsEnd;
	map<string, int> m_mapNiEnd;
	map<string, int> m_mapNzEnd;

    set<string> m_setNiNsNzsuf;
	set<string> m_setNhpresuf;
	set<string> m_setNotTime;
	set<string> m_setNm;
	set<string> m_setNotNm;
	set<string> m_setNibeg;
	set<string> m_setNiAbb;
	set<string> m_setNsAbb;

	map<string, RULEIDXNODE> m_mapRuleIdx;

	int m_OOVWordNum;
	PROBNODE* pProb;

private:
	enum
	{
		PROBRULENum = 3570,
		NsendNum = 1,
        NiendNum = 2,
		NzendNum = 3,
	};	
};

#endif


