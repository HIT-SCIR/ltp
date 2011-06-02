#include "InitDic.h"
#include "MyNLPlib.h"
#include <fstream>
#include <string.h>

using namespace std;

InitDic::InitDic()
{
	pProb = new PROBNODE [PROBRULENum];
}

InitDic::~InitDic()
{
	delete [] pProb;
}

void InitDic::releaseRes()
{
	for (int i=0; i<PROBRULENum; ++i)
	{
		delete [] pProb[i].probrule;
	}
	//pProb = NULL;	
}

///////////////////////////////////////////////////////////////
//	函 数 名 : loadObver
//	所属类名 : InitDic
//	函数功能 : 加载HMM的观察值
//	处理过程 : 
//	备    注 : 
//	作    者 : taozi
//	时    间 : 2006年5月24日
//	返 回 值 : int
//	参数说明 : char* infile
///////////////////////////////////////////////////////////////
int InitDic::loadObver(char* infile)
{
	string path(infile);
	path += "\\ObservationList.txt";
	ifstream Obser(path.c_str());
    if (! infile)
	{
		return 0;
	}
	
	string strin;
	int lineNum = 0;
	while (getline(Obser, strin))
	{
		if (! strin.empty())
		{
			m_mapObserstr2int.insert(map<string, int>::value_type(strin, lineNum));
			//m_mapObserint2str.insert(map<int, string>::value_type(lineNum, strin));
			++lineNum;
		}
	}
	m_OOVWordNum = lineNum;
	Obser.close();
	return 1;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : loadState
//	所属类名 : InitDic
//	函数功能 : 加载HMM的状态标记
//	处理过程 : 
//	备    注 : 
//	作    者 : taozi
//	时    间 : 2006年5月24日
//	返 回 值 : int
//	参数说明 : char* infile
///////////////////////////////////////////////////////////////
int InitDic::loadState(char* infile)
{
	string path(infile);
	path += "\\statelist.txt";
	ifstream State(path.c_str());
    if (! infile)
	{
		return 0;
	}
	
	string strin;
	int lineNum = 0;
	while (getline(State, strin))
	{
		if (! strin.empty())
		{
			m_mapStatestr2int.insert(map<string, int>::value_type(strin, lineNum));
			m_mapStateint2str.insert(map<int, string>::value_type(lineNum, strin));
			++lineNum;
		}
	}

	State.close();
	return 1;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : loadRule
//	所属类名 : InitDic
//	函数功能 : 加载所有的规则
//	处理过程 : 
//	备    注 : 
//	作    者 : taozi
//	时    间 : 2006年5月30日
//	返 回 值 : int
//	参数说明 : char* infile：存放规则文件名的列表的文件名
///////////////////////////////////////////////////////////////
int InitDic::loadRule(char* infile)
{
	string path(infile);
	path += "\\rulefilelist.txt";
	ifstream in(path.c_str());
	if (! in)
	{
		return 0;
	}

	string strin;
	string file;
	int fileIndex;
	//ifstream inrule;
	int fileNum = 0;
	int flag = 0;
	while (in >> strin)
	{
		in >> file;
		fileIndex = atoi(file.c_str());
		switch (fileIndex)
		{
		case 0: flag = loadProrule((char*)((string)infile+strin).c_str());
			    break;

		case 1: flag = addRuletoMap((char*)((string)infile+strin).c_str(), m_mapNsEnd, NsendNum); //地名后缀词规则
			    break;

		case 2: flag = addRuletoMap((char*)((string)infile+strin).c_str(), m_mapNiEnd, NiendNum); //机构名后缀词规则
			    break;

		case 3: flag = addRuletoMap((char*)((string)infile+strin).c_str(), m_mapNzEnd, NzendNum); //专有名词后缀词规则
			    break;

		case 4: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNiNsNzsuf); //Ni、Ns、Nz后边词
			    break;
				
		case 5: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNhpresuf); //Nh的前边词和后边词
			    break;
				
		case 6: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNotTime); //非时间词
			    break;
				
		case 7: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNm); //数词
			    break;

		case 8: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNotNm); //非数词
			    break;
		case 9: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNibeg); //机构名触发词
			    break;
        case 10: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNiAbb); //机构名简称词
			     break;
		case 11: flag = addRuletoSet((char*)((string)infile+strin).c_str(), m_setNsAbb); //地名简称词
			     break;

		default: cout << "规则文件的编号有误!" << endl;
			     break;
		}

		if (!flag)
		{
			cerr << "加载规则文件有误!" << endl;
			return 0;
		}
	}

	return 1;
}


///////////////////////////////////////////////////////////////
//	函 数 名 : loadProrule
//	所属类名 : InitDic
//	函数功能 : 加载词性串规则
//	处理过程 : 
//	备    注 : 
//	作    者 : taozi
//	时    间 : 2006年6月5日
//	返 回 值 : int
//	参数说明 : char* infile
///////////////////////////////////////////////////////////////
int InitDic::loadProrule(char* infile)
{
	ifstream in(infile);
	if (! in)
	{
		return 0;
	}

	string strin;
	string probrule;
	string probability;
	string::size_type pos1 = 0;
	string::size_type pos2 = 0;
	int lineNum = 0;
	
	while (getline(in, strin))
	{
		if (! strin.empty())
		{
			memset(&pProb[lineNum], 0, sizeof(PROBNODE));
			pos1 = strin.find("#", 0);
			probrule = strin.substr(0, pos1);
			int len = probrule.size();
            pProb[lineNum].probrule = new char [len+1];	            

			strncpy(pProb[lineNum].probrule, probrule.c_str(), len);
			pProb[lineNum].probrule[len] = '\0';
			setRuleindex(pProb[lineNum].probrule, lineNum);
            
			pos2 = strin.find("*", pos1);
            probability = strin.substr(pos1+1, pos2-pos1-2);
			pProb[lineNum].probability = atof(probability.c_str());

			pProb[lineNum].NEtypeNum = atoi(strin.substr(pos2+1).c_str());

			++lineNum;
		}
	}
	in.close();
	return 1;
}


void InitDic::setRuleindex(string strPosRule, int index)
{
	RULEIDXNODE RuleIdxNode;
	memset(&RuleIdxNode, 0, sizeof(RULEIDXNODE));
	string strtemp;

	int pos = strPosRule.find(" ", 0);
	pos = strPosRule.find(" ", pos+1);
	strtemp = strPosRule.substr(0, pos);

	if (m_mapRuleIdx.find(strtemp) == m_mapRuleIdx.end())
	{
		RuleIdxNode.begIndex = index;
        RuleIdxNode.endIndex = index;
		m_mapRuleIdx[strtemp] = RuleIdxNode;
	}
	else
	{
		if (m_mapRuleIdx[strtemp].endIndex < index)
		{
			m_mapRuleIdx[strtemp].endIndex = index;
		}
	}
}
///////////////////////////////////////////////////////////////
//	函 数 名 : addRuletoMap
//	所属类名 : InitDic
//	函数功能 : 将规则加入到map中
//	处理过程 : 
//	备    注 : 
//	作    者 : 
//	时    间 : 2006年6月22日
//	返 回 值 : int
//	参数说明 : char* infile,
//				 map<string,
//				 int>& mapName,
//				 int index
///////////////////////////////////////////////////////////////

int InitDic::addRuletoMap(char* infile, map<string, int>& mapName, int index)
{
	ifstream in(infile);
	if (! in)
	{
		return 0;
	}

   string strin;
   while (getline(in, strin))
   {
	   if (! strin.empty())
	   {
		   mapName[strin] = index;
	   }
   }

   in.close();
   return 1;
}


int InitDic::getWordIndexInMap(map<string, int>& mapName, string& word)
{
	if (mapName.find(word) != mapName.end())
	{
		return mapName[word];
	}
	else
	{
		return -1;
	}
}

void InitDic::getRuleIndex(string& strRule, int& begIdx, int& endIdx)
{
	if (m_mapRuleIdx.find(strRule) != m_mapRuleIdx.end())
	{
		begIdx = m_mapRuleIdx[strRule].begIndex;
		endIdx = m_mapRuleIdx[strRule].endIndex;
	}
	else
	{
		begIdx = 0;
		endIdx = 0;
	}
}


int InitDic::addRuletoSet(char* infile, set<string>& setName)
{
	ifstream in(infile);
	if (! in)
	{
		return 0;
	}

   string strin;
   while (getline(in, strin))
   {
	   if (! strin.empty())
	   {
		   setName.insert(set<string>::value_type(strin)); // [strin] = index;
	   }
   }

   in.close();
   return 1;
}



int InitDic::getObserIndex(string& obser)
{
	if (m_mapObserstr2int.find(obser) != m_mapObserstr2int.end())
	{
		return m_mapObserstr2int[obser];
	}
	else
	{
		return m_OOVWordNum;
	}
}

int InitDic::getStateIndex(string& state)
{
	if (m_mapStatestr2int.find(state) != m_mapStatestr2int.end())
	{
		return m_mapStatestr2int[state];
	}
	else
	{
		return m_mapStatestr2int["O"];
	}
}

string InitDic::getStatestr(int stateIndex)
{
	if (m_mapStateint2str.find(stateIndex) != m_mapStateint2str.end())
	{
		return m_mapStateint2str[stateIndex];
	}
	else
	{
		return string("O");
	}
}
