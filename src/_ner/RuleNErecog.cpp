#include "RuleNErecog.h"
#include <string.h>

extern int g_isEntity;
extern int g_isTime;
extern int g_isNum;

Ruletest::Ruletest()
{
	memset(&Rulenode, 0, sizeof(RULENODE));
}

void Ruletest::setObject(InitDic* dic)
{
	m_pdic = dic;
}

void Ruletest::RuleNErecog(vector<string>& vecNE,
						   vector< pair<string, string> >& vecpaSen)
{
	unsigned int pos1 = 0;
	unsigned int pos2 = vecNE.size() - 1;
	string NEtag;

	for (int i = 0, j=vecNE.size()-1; i < vecpaSen.size(); i++,j--)
	{
		// 修复出现 “……O I-Ni……” 情况，例如：央视/j#O 情/n#E-Ni 系/n#O 帅乡/nh#B-Nh ·/wp#I-Nh 江津/nh#E-Nh 群星/n#O 演唱会/n#O
		if (vecNE[j].at(0) == 'I' && (j == vecNE.size() - 1 || vecNE[j + 1] == "O"))
		{
			vecNE[j] = "O";
		}
	}
		
	while (pos1 < vecpaSen.size())
	{
		NEtag = vecNE[pos2];

		if (g_isTime && vecpaSen[pos1].second == "nt")//Time
		{
			recogNtNr(vecpaSen, pos1, vecNE);
		}
		else if (g_isNum && vecpaSen[pos1].second == "m")//Num
		{
			recogNm(vecpaSen, pos1, vecNE);
		}
		else if (g_isEntity && vecpaSen[pos1].second == "nz" && NEtag == "O")//Proper noun
		{
			vecNE[pos2] = "B-Nz";
			//++pos1;
			//--pos2;
		}
		else if (g_isEntity && vecpaSen[pos1].second == "ns" && pos2 > 0 && pos1 > 0
				&& vecpaSen[pos1 + 1].second != "ni" 
				&& vecpaSen[pos1 + 1].second != "nz"
				&& vecpaSen[pos1 - 1].second != "ns"
				&& vecpaSen[pos1 + 1].second != "ns"
				&& vecpaSen[pos1 + 1].second != "j" && NEtag == "O")//address
		{
			vecNE[pos2] = "B-Ns";
			//++pos1;
			//--pos2;
		}
		else if (g_isEntity && NEtag != "O")
		{
			recogComplexNE(vecpaSen, pos1, vecNE);
		}
		else if (g_isEntity &&
			m_pdic->m_setNibeg.find(vecpaSen[pos1].first) != m_pdic->m_setNibeg.end())//若是机构名的起始词
		{
			recogMissedNi(vecpaSen, pos1, vecNE);
		}
		else if (g_isEntity && 
			((m_pdic->m_setNiAbb.find(vecpaSen[pos1].first) != m_pdic->m_setNiAbb.end()) ||
			(m_pdic->m_setNsAbb.find(vecpaSen[pos1].first) != m_pdic->m_setNsAbb.end())) &&
			vecpaSen[pos1].second=="j")

		{
			recogAbbreviation(vecpaSen, pos1, vecNE);
		}
		
		else
		{
			++pos1;
			--pos2;
		}
		pos2 = vecNE.size() - 1 - pos1;
	}	
}



void Ruletest::recogAbbreviation(vector< pair<string, string> >& vecpaSen, 
		                         unsigned int& pos, vector<string>& vecNE)
{
	if (m_pdic->m_setNiAbb.find(vecpaSen[pos].first) != m_pdic->m_setNiAbb.end())
	{
		addNETag(pos, 1, string("Ni"), vecNE);
		++pos;
	}
	else if (m_pdic->m_setNsAbb.find(vecpaSen[pos].first) != m_pdic->m_setNsAbb.end())
	{
		addNETag(pos, 1, string("Ns"), vecNE);
		++pos;
	}
}

void Ruletest::recogMissedNi(vector< pair<string, string> >& vecpaSen, 
		                     unsigned int& pos, vector<string>& vecNE)
{
	int NEpos = vecNE.size() - 1 - pos;
	int niEndidx = findNsNiNzendWord(pos, vecpaSen, NiendNum);
	if (niEndidx != 0)
	{
		unsigned int NEniEndidx =  vecNE.size() - 1 - niEndidx;
		if (vecNE[NEniEndidx].at(0)=='B' || vecNE[NEniEndidx].at(0)=='I')
		{
			++pos;
			return;
		}
		correctNsNiNztags(NEpos, NEniEndidx, NiendNum, vecNE);
		pos = niEndidx + 1;
	}
	else
	{
		++pos;
	}
}

void Ruletest::recogComplexNE(vector< pair<string, string> >& vecpaSen, 
							  unsigned int& pos, vector<string>& vecNE)
{
	unsigned int NEpos = vecNE.size() - 1 - pos;
	string NEtag = vecNE[NEpos];
	if ( NEpos == 0)
	{
		NEpos--;
		pos++;
		return;
	}

	if ((NEtag=="B-Nh" || NEtag=="B-Ns" ||
		 NEtag=="B-Ni" || NEtag=="B-Nz") && 
		(vecNE[NEpos-1]=="O" || vecNE[NEpos-1].at(0) == 'B'))
	{
		if (match_SingleNE(vecpaSen, pos, NEtag.substr(2))) //当确认是一个简单NE时返回值为1
		{
			if (vecpaSen[pos].second == "nz")
			{
				vecNE[NEpos] = "B-Nz";
			}
			//if ((NEtag=="B-Ns"||NEtag=="B-Nh") && findNsNiNzendWord(pos, vecpaSen, NzendNum))
			//{
			//	vecNE[NEpos--] = "B-Nz";
			//	vecNE[NEpos] ="I-Nz";
			//	pos++;
			//}
			++pos;
			--NEpos;
		}
		else if (match_ComplexNE(vecpaSen, pos))
		{
			correctNEResult(vecNE);
			NEpos = vecNE.size() - 2 - Rulenode.nRuleEnd;
			pos = Rulenode.nRuleEnd + 1;
		}
		else
		{
			NEpos--;
			pos++;
		}
	}
	else if (NEtag=="B-Ni")
	{	
		amendComplexNi(vecpaSen, pos, vecNE);
				
	}
	else if (NEtag == "B-Ns")
	{
		amendComplexNs(vecpaSen, pos, vecNE);
	}			
	else if (NEtag == "B-Nz")
	{
		amendComplexNz(vecpaSen, pos, vecNE);
	}	
	else
	{
		++pos;
	}
}

void Ruletest::amendComplexNi(vector< pair<string, string> >& vecpaSen, 
							  unsigned int& pos, vector<string>& vecNE)
{
	int NEpos = vecNE.size() - 1 - pos;
	string NEtemp = vecNE[NEpos];
	int NEendIdx = findNEend(NEpos, vecNE);
	int sEndIdx = pos + NEpos - NEendIdx;
	int niEndidx = 0;
	if (NEendIdx != 0 &&
		m_pdic->getWordIndexInMap(m_pdic->m_mapNiEnd, vecpaSen[sEndIdx].first) != NiendNum)
	{
		correctNsNiNztags(NEpos, NEendIdx, 0, vecNE);
		niEndidx = findNsNiNzendWord(sEndIdx, vecpaSen, NiendNum);
		
		if (niEndidx != 0)
		{
			unsigned int NEniEndidx =  vecNE.size() - 1 - niEndidx;
			correctNsNiNztags(NEpos, NEniEndidx, NiendNum, vecNE);

			pos = niEndidx + 1;
		}
		else if (match_ComplexNE(vecpaSen, pos))
		{
			correctNEResult(vecNE);				
			pos = Rulenode.nRuleEnd + 1;
		}		
	}
	else if (NEendIdx != 0)
	{
		pos = sEndIdx + 1;
	}
	else
	{
		correctNsNiNztags(NEpos, 0, 0, vecNE);
		pos = pos + 1;
	}
}


void Ruletest::amendComplexNs(vector< pair<string, string> >& vecpaSen, 
							  unsigned int& pos, vector<string>& vecNE)
{
	int NEpos = vecNE.size() - 1 - pos;
	int NEendIdx = findNEend(NEpos, vecNE);
	int sEndIdx = pos + NEpos - NEendIdx;
	int nsEndidx = 0;
	if (NEendIdx > 0 && 
        m_pdic->m_setNiNsNzsuf.find(vecpaSen[sEndIdx+1].first) != m_pdic->m_setNiNsNzsuf.end())
	{
		pos = sEndIdx + 1;
	}
	else if (NEendIdx != 0)
	{
		nsEndidx = findNsNiNzendWord(sEndIdx, vecpaSen, NsendNum);				
		if (nsEndidx != 0)
		{
			unsigned int NEniEndidx =  vecNE.size() - 1 - nsEndidx;
			correctNsNiNztags(NEpos, NEendIdx, 0, vecNE);
			correctNsNiNztags(NEpos, NEniEndidx, NsendNum, vecNE);
			pos = nsEndidx + 1;
		}
		else if (match_ComplexNE(vecpaSen, pos))
		{
			correctNsNiNztags(NEpos, NEendIdx, 0, vecNE);
			correctNEResult(vecNE);				
			pos = Rulenode.nRuleEnd + 1;
		}
		else
		{
			pos = sEndIdx + 1;
		}
	}
	else
	{
//		correctNsNiNztags(NEpos, 0, 0, vecNE);
		pos = pos + 1;
	}
}

void Ruletest::amendComplexNz(vector< pair<string, string> >& vecpaSen, 
							  unsigned int& pos, vector<string>& vecNE)
{
	int NEpos = vecNE.size() - 1 - pos;
	int NEendIdx = findNEend(NEpos, vecNE);
	int sEndIdx = pos + NEpos - NEendIdx;
	int nsEndidx = 0;
	if (NEendIdx > 0 && 
		m_pdic->m_setNiNsNzsuf.find(vecpaSen[sEndIdx+1].first) != m_pdic->m_setNiNsNzsuf.end())
	{
		pos = sEndIdx + 1;
	}
	else if (NEendIdx != 0)
	{
		nsEndidx = findNsNiNzendWord(sEndIdx, vecpaSen, NzendNum);				
		if (nsEndidx != 0)
		{
			unsigned int NEniEndidx =  vecNE.size() - 1 - nsEndidx;
			correctNsNiNztags(NEpos, NEendIdx, 0, vecNE);
			correctNsNiNztags(NEpos, NEniEndidx, NzendNum, vecNE);
			pos = nsEndidx + 1;
		}
		else if (match_ComplexNE(vecpaSen, pos))
		{
			correctNsNiNztags(NEpos, NEendIdx, 0, vecNE);
			correctNEResult(vecNE);				
			pos = Rulenode.nRuleEnd + 1;
		}
		else
		{
			pos = sEndIdx + 1;
		}
	}
	else
	{
		correctNsNiNztags(NEpos, 0, 0, vecNE);
		pos = pos + 1;
	}
}


int Ruletest::findNsNiNzendWord(unsigned int begpos,
								vector< pair<string, string> >& vecpaSen, int typeNum)
{
	unsigned int pos = begpos + 1;
	int num = 0;
	while (pos<vecpaSen.size() && num<Maxchecklen)
	{
		if (m_pdic->m_setNiNsNzsuf.find(vecpaSen[pos].first) != m_pdic->m_setNiNsNzsuf.end())
		{
			return 0;
		}
		switch(typeNum)
		{
		case NsendNum: if (m_pdic->getWordIndexInMap(m_pdic->m_mapNsEnd, vecpaSen[pos].first) == NsendNum)
					   {
						   return pos;
					   }
					   ++num;
					   ++pos;
					   break;
		case NiendNum: if (m_pdic->getWordIndexInMap(m_pdic->m_mapNiEnd, vecpaSen[pos].first) == NiendNum)
					   {
						   return pos;
					   }
					   ++num;
					   ++pos;
					   break;
		case NzendNum: if (m_pdic->getWordIndexInMap(m_pdic->m_mapNzEnd, vecpaSen[pos].first) == NzendNum)
					   {
						   return pos;
					   }
					   ++num;
					   ++pos;
					   break;
		default: return 0;
		}		
		
	}
	return 0;
}

int Ruletest::findNEend(unsigned int begpos, vector<string>& vecNE)
{
	int pos = (int)begpos - 1;
	while (pos >= 0)
	{
		if (vecNE[pos] == "O" || vecNE[pos].at(0) == 'B')
		{
			if (pos == begpos-1)
			{
				return pos;
			}
			return pos + 1;
		}
		--pos;	
	}
	return 0;
}


void Ruletest::recogNtNr(vector< pair<string, string> >& vecpaSen,
						 unsigned int& pos, vector<string>& vecOut)
{
	int nLength =0;
	int nbegin = 0;
	int statusNum = 0;
//	bool timeflag = false;

	if ( m_pdic->m_setNotTime.find(vecpaSen[pos].first) != m_pdic->m_setNotTime.end())
	{
		++pos;
	}
    else
	{
		nbegin = pos;
		while ((pos<(int)vecpaSen.size())&& (vecpaSen[pos].second=="nt"))
		{
			switch(statusNum)
			{
			case 0: if ((vecpaSen[pos].first.find("年", 0) != -1) ||
			        	(vecpaSen[pos].first.find("月", 0) != -1) ||
			        	(vecpaSen[pos].first.find("日", 0) != -1)) 
					{
						statusNum = 1;
					}
				    else if ((vecpaSen[pos].first.find("凌晨", 0) != -1) ||
							 (vecpaSen[pos].first.find("早晨", 0) != -1) ||
							 (vecpaSen[pos].first.find("早上", 0) != -1) ||
							 (vecpaSen[pos].first.find("上午", 0) != -1) ||
							 (vecpaSen[pos].first.find("中午", 0) != -1) ||
							 (vecpaSen[pos].first.find("晌午", 0) != -1) ||
							 (vecpaSen[pos].first.find("下午", 0) != -1) ||
							 (vecpaSen[pos].first.find("傍晚", 0) != -1) ||
							 (vecpaSen[pos].first.find("黄昏", 0) != -1) ||
							 (vecpaSen[pos].first.find("晚上", 0) != -1) ||
							 (vecpaSen[pos].first.find("夜里", 0) != -1) ||
							 (vecpaSen[pos].first.find("半夜", 0) != -1) ||
							 (vecpaSen[pos].first.find("深夜", 0) != -1))
					{
						statusNum = 2;
					}
					else if ((vecpaSen[pos].first.find("时", 0) != -1)||
							(vecpaSen[pos].first.find("点钟", 0) != -1)||
							(vecpaSen[pos].first.find("点半", 0) != -1))
					{
						statusNum = 2;
					}
					else if ((vecpaSen[pos].first.find("分", 0) != -1)||
							 (vecpaSen[pos].first.find("秒", 0) != -1))
					{
						statusNum = 3;
					}
					++nLength;
					++pos;
					break;

			case 1: if ((vecpaSen[pos].first.find("年", 0) != -1) ||
			        	(vecpaSen[pos].first.find("月", 0) != -1) ||
			        	(vecpaSen[pos].first.find("日", 0) != -1)) 
					{
						statusNum = 1;	
					}
				    else if ((vecpaSen[pos].first.find("时", 0) != -1) ||
							 (vecpaSen[pos].first.find("点", 0) != -1) ||
						     (vecpaSen[pos].first.find("分", 0) != -1))
					{
						statusNum = 2;
					}
					++nLength;
					++pos;
					break;
					
			case 2: if ((vecpaSen[pos].first.find("分", 0) != -1) ||
						(vecpaSen[pos].first.find("秒", 0) != -1) ||
						(m_pdic->m_setNotTime.find(vecpaSen[pos].first) == m_pdic->m_setNotTime.end()))
					{
						statusNum = 2;
					}
					else if ((vecpaSen[pos].first.find("凌晨", 0) != -1) ||
						(vecpaSen[pos].first.find("早晨", 0) != -1) ||
						(vecpaSen[pos].first.find("早上", 0) != -1) ||
						(vecpaSen[pos].first.find("上午", 0) != -1) ||
						(vecpaSen[pos].first.find("中午", 0) != -1) ||
						(vecpaSen[pos].first.find("晌午", 0) != -1) ||
						(vecpaSen[pos].first.find("下午", 0) != -1) ||
						(vecpaSen[pos].first.find("傍晚", 0) != -1) ||
						(vecpaSen[pos].first.find("黄昏", 0) != -1) ||
						(vecpaSen[pos].first.find("晚上", 0) != -1) ||
						(vecpaSen[pos].first.find("夜里", 0) != -1) ||
						(vecpaSen[pos].first.find("半夜", 0) != -1) ||
						(vecpaSen[pos].first.find("深夜", 0) != -1))
					{
						statusNum = 2;
					}
					++nLength;
				    ++pos;
					break;

			default: break;
			}//switch
			if (statusNum == 3)
			{
				break;
			}
		}//while
        
		if (statusNum == 1)
		{
			addNETag(nbegin, nLength, string("Nr"), vecOut);
		}
		else if (statusNum == 2)
		{
			addNETag(nbegin, nLength, string("Nt"), vecOut);
		}
		else if (statusNum == 3)
		{
			addNETag(nbegin, nLength, string("Nm"), vecOut);
		}
	}
}


void Ruletest::recogNm(vector< pair<string, string> >& vecpaSen,
		               unsigned int& pos, vector<string>& vecOut)
{
	int nbegin = pos;
	int nLength = 0; //记录数字表达式的长度
	int flagCad = 0; //候选数词
	if (m_pdic->m_setNotNm.find(vecpaSen[pos].first) != m_pdic->m_setNotNm.end())
	{
		pos++;
	}
	else
	{
		while ((pos<(int)vecpaSen.size()) && ((vecpaSen[pos].second == "m")||(vecpaSen[pos].second == "q")))
		{
			++nLength;
			++pos;
			if ((pos<(int)vecpaSen.size()) && (vecpaSen[pos].first == "."))
			{
				flagCad = 1; //标记为候选，主要处理7/m ./wp 2级/m的形式
				// nLength++;
				++pos;
			}
			if (flagCad == 1)
			{
				++nLength;
				flagCad = 0;
			}
		}
		if ((pos<(int)vecpaSen.size()) &&
			(m_pdic->m_setNm.find(vecpaSen[pos].first)!=m_pdic->m_setNm.end()) &&
			(vecpaSen[pos].second != "v")) //将后边的词也加入
		{
			++nLength;
			++pos;
		}
		else if ((pos<(int)vecpaSen.size()) &&
			    (m_pdic->m_setNotNm.find(vecpaSen[pos].first) != m_pdic->m_setNotNm.end())) //后边是停用词或边界词
		{
	        ++pos;
		}
		else if (pos<(int)vecpaSen.size() && vecpaSen[pos].first=="年代")
		{
			addNETag(nbegin, 2, string("Nr"), vecOut);
			++pos;
			return;
		}
		else
		{
			//++pos;
			return;
		}

		//addNETag(nbegin, nLength, string("Nm"), vecOut);	
	}
}


void Ruletest::addNETag(int begin, int len, string NEtype, vector<string>& vecOut)
{
	int NEbeg = vecOut.size() - 1 - begin;
	if (len == 1)
	{
		vecOut[NEbeg] = "B-" + NEtype; //S-Nm
	}
	else
	{
		vecOut[NEbeg] = "B-" + NEtype;
		for (int i=NEbeg-1; i > NEbeg-len; i--)
		{
			vecOut[i] = "I-" + NEtype;
		};
	}
}
///////////////////////////////////////////////////////////////
//	函 数 名 : correctNEResult
//	所属类名 : Ruletest
//	函数功能 : 
//	处理过程 : 
//	备    注 : 
//	作    者 : 
//	时    间 : 2006年6月27日
//	返 回 值 : void
//	参数说明 : vector<string>& vecNE
///////////////////////////////////////////////////////////////
void Ruletest::correctNEResult(vector<string>& vecNE)
{
	string NEtype = getNEtype(Rulenode.nRuleType);
	int vecNELen = vecNE.size();
	int NERuleBeg = vecNELen - 1 - Rulenode.nRuleBeg;
	int NERuleEnd = vecNELen - 1 - Rulenode.nRuleEnd;

	vecNE[NERuleBeg] = "B-" + NEtype;
	for (int i=NERuleBeg - 1; i >= NERuleEnd; --i)
	{
		vecNE[i] = "I-" + NEtype;
	}
}

void Ruletest::correctNsNiNztags(unsigned int begpos,
								 unsigned int endpos,
								 int typeNum,
								 vector<string>& vecNE)
{
	int i;
	string NEB_tag;
	string NEI_tag;
	switch(typeNum)
	{
	case NsendNum: NEB_tag = "B-Ns";
			   	   NEI_tag = "I-Ns";
		           break;
	case NiendNum: NEB_tag = "B-Ni";
				   NEI_tag = "I-Ni";
				   break;
	case NzendNum: NEB_tag = "B-Nz";
				   NEI_tag = "I-Nz";
				   break;
	default:NEB_tag = "O";
			NEI_tag = "O";
	}

	vecNE[begpos] = NEB_tag;

	for (i=begpos-1; i>=endpos; --i)
	{
		if (i>=0)
		{
			vecNE[i] = NEI_tag;
		}
		else 
			break;	
	}


}


string Ruletest::getNEtype(int NEtypeNum)
{
	switch (NEtypeNum)
	{
	case 1: return "Ns";
	case 2: return "Ni";
	case 3: return "Nz";
	default: return "";
	}

}

int Ruletest::match_ComplexNE(vector< pair<string, string> >& vecpaSen, unsigned int pos)
{
	string strPosRule; //存放词性串规则
	int index = 0;
	int begIndex = 0;
	int endIndex = 0;
	string strPos;
	unsigned int i;
	for (i=pos; i<pos+MaxRuleLen; ++i)  //抽取词性串
	{
		if (i<vecpaSen.size())
		{
			strPosRule += vecpaSen[i].second;
			strPosRule += " ";
		}
		else
		{
			break;
		}
	}

	int flag = 0;
	string strWord;
	int NsEnd = 0;
	int NiEnd = 0;
	int NzEnd = 0;

	memset(&Rulenode, 0, sizeof(RULENODE));

   if (pos+1 < vecpaSen.size())
   {
	   strPos += vecpaSen[pos].second + " ";
	   strPos += vecpaSen[pos+1].second;
	   m_pdic->getRuleIndex(strPos, begIndex, endIndex);
	   
	   int posNE = 0;       
	   if (begIndex>0 && begIndex<PROBRULENum &&
		   endIndex>0 && endIndex<PROBRULENum &&
		   begIndex <= endIndex)
	   {
		   for (i=begIndex; i<=endIndex; ++i)
		   {
			   int num = 0;
			   if (isRule(strPosRule, m_pdic->pProb[i].probrule, num) >= 0)
			   {
				   posNE = pos + num - 1;
				   
				   NsEnd = m_pdic->getWordIndexInMap(m_pdic->m_mapNsEnd, vecpaSen[posNE].first); //m_pdic->m_mapNsEnd[strWord];
				   NiEnd = m_pdic->getWordIndexInMap(m_pdic->m_mapNiEnd, vecpaSen[posNE].first);//m_pdic->m_mapNiEnd[strWord];
				   NzEnd = m_pdic->getWordIndexInMap(m_pdic->m_mapNzEnd, vecpaSen[posNE].first);//m_pdic->m_mapNzEnd[strWord];
				   if (NsEnd == m_pdic->pProb[i].NEtypeNum ||
					   NiEnd == m_pdic->pProb[i].NEtypeNum ||
					   NzEnd == m_pdic->pProb[i].NEtypeNum)
				   {
					   if ((num > Rulenode.nRuleLen) || (num==Rulenode.nRuleLen &&
						   m_pdic->pProb[i].probability>Rulenode.nRulePro))
					   {
						   flag = 1;
						   Rulenode.nRuleBeg = pos;
						   Rulenode.nRuleEnd = posNE;
						   Rulenode.nRuleLen = num;
						   Rulenode.nRuleType = m_pdic->pProb[i].NEtypeNum;
						   Rulenode.nRulePro = m_pdic->pProb[i].probability;
					   }
				   }
			   }		  
		   }
	   }
	   strPosRule.erase();
	   strPos.erase();
	   return flag;
   } 
   return 0;
}

int Ruletest::isRule(string& strPos, char* pRule, int& num)
{
	num = 0;
	unsigned int posS = 0;
	unsigned int posR = 0;
	while(posS != strPos.size() && posR != strlen(pRule))
	{
		if(strPos[posS] == pRule[posR])
		{
			if (strPos[posS] == ' ')
			{
				++num;
			}
			posS ++;
			posR ++;
		}
		else
		{
			return -1;
		}
	}

	if(posR == strlen(pRule) && posS < strPos.size())  //如果有匹配部分，即flag=1
		return 1;
	else if(posS == strlen(pRule) && posR == strPos.size())
		return 0;
	else return -1;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : match_SingleNE
//	所属类名 : Ruletest
//	函数功能 : 匹配独立的NE
//	处理过程 : 
//	备    注 : 
//	作    者 : 
//	时    间 : 2006年6月22日
//	返 回 值 : int
//	参数说明 : vector<pair<string,
//				 string> >& vecpaSen,
//				          int pos,
//				 string& NEtype
///////////////////////////////////////////////////////////////
int Ruletest::match_SingleNE(vector< pair<string, string> >& vecpaSen,
							 unsigned int pos, string NEtype)
{
	string word;
	int comp1 = 0;
	int comp2 = 0;

	if (NEtype=="Ni" || NEtype=="Ns" || NEtype=="Nz")
	{
		if (pos < vecpaSen.size()-1)
		{
			word = vecpaSen[pos+1].first;
			if (m_pdic->m_setNiNsNzsuf.find(word) != m_pdic->m_setNiNsNzsuf.end())
			{
				comp1 = 1;
			}		
		}
		else
		{
			comp1 = 1;
		}
	}
	else if (NEtype=="Nh")
	{
		if (pos < vecpaSen.size()-1)
		{
			word = vecpaSen[pos+1].first;
            if (m_pdic->m_setNhpresuf.find(word) != m_pdic->m_setNhpresuf.end())
			{
				comp1 = 1;
			}
		}

		if (pos > 0)
		{
			word = vecpaSen[pos-1].first;
			if (m_pdic->m_setNhpresuf.find(word) != m_pdic->m_setNhpresuf.end())
			{
				comp2 = 1;
			}
		}
		
		if (pos==0 || pos+1==vecpaSen.size())
		{
			comp1 = 1;
		}
	}

	if (comp1 || comp2)
	{
		return 1;
	}
	else 
	{
		return 0;
	}
}
