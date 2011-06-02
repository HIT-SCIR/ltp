#include "SRLBaseline.h"

///////////////////////////////////////////////////////////////
//	Function Name 	: SRLBaseline
//	Belong to Class : SRLBaselin
//	Function  	: 
//	Processing 	: 
//	Remark 		: 
//	Author 		: Frumes
//	Time 		: 2007年1月4日
//	Return Value 	: 
//	Parameter Comment : 
///////////////////////////////////////////////////////////////
SRLBaseline::SRLBaseline(string configXml, string selectFeats)
{
	
}
///////////////////////////////////////////////////////////////
//	函 数 名 : ~SRLBaseline
//	所属类名 : SRLBaselin
//	函数功能 : The Class Destructor
//	处理过程 : 
//	备    注 : 
//	作    者 : hjliu
//	时    间 : 2006年6月21日
//	返 回 值 : 
//	参数说明 : 
///////////////////////////////////////////////////////////////
SRLBaseline::~SRLBaseline()
{
}

///////////////////////////////////////////////////////////////
//	函 数 名 : IsFilter
//	所属类名 : SRLBaseline
//	函数功能 : Check if the node will be filtered: only when the node 
//			   is predicate and punctation
//	处理过程 : 
//	备    注 : 
//	作    者 : hjliu
//	时    间 : 2006年7月14日
//	返 回 值 : void
//	参数说明 : const int nodeID
///////////////////////////////////////////////////////////////
inline bool SRLBaseline::IsFilter(int nodeID, 
								  int intCurPd) const
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


//for now used
///////////////////////////////////////////////////////////////
//	Function Name 	: setPredicate
//	Belong to Class : SRLBaseline
//	Function  	: 
//	Processing 	: 
//	Remark 		: 
//	Author 		: Frumes
//	Time 		: 2007年1月5日
//	Return Value 	: void
//	Parameter Comment : const vector<int>& vecPred
///////////////////////////////////////////////////////////////
void SRLBaseline::SetPredicate(const vector<int>& vecPred)
{
	m_vecPredicate = vecPred;
}

void SRLBaseline::setDataPreProc(const DataPreProcess* dataPreProc)
{
	m_dataPreProc = dataPreProc;
}