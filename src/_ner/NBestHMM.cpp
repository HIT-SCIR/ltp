/**************************************************************
	文 件 名 : NBestHMM.cpp
	文件功能 : 包含此模块所有函数的实现
	作    者 : Truman
	创建时间 : 2003年10月25日
	项目名称 : 隐马尔可夫模型N条最优路径搜索通用算法模块
	编译环境 : Visual C++ 6.0
	备    注 : 
	历史记录 :  
**************************************************************/
//#include "StdAfx.h"
#include "NBestHMM.h"

PathNode::PathNode()
{
	accuProb = 0;	//路径累计权值置0
	preStateIndex = -1;	//-1表示前一个节点的状态为空
	preNode = NULL;	
	next = NULL;
	prev = NULL;
	curStateIndex = -1;
}

PathNode::~PathNode()
{
}

CNBestHMM::CNBestHMM()
{
// 	headPath = NULL;
}

CNBestHMM::~CNBestHMM()
{
//	PathNode *p = headPath;
//	while(p != NULL)
//	{
//		PathNode *p1 = p;
//		p = p->next;
//		delete p1;
//	}
//	headPath = NULL;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : Initialize
//	所属类名 : CNBestHMM
//	函数功能 : 对CNBestHMM类进行初始化，如果初始化失败则返回false
//	备    注 : 
//	作    者 : Truman
//	时    间 : 2003年10月26日
//	返 回 值 : bool
//	参数说明 : const string &startFile,
//				          const string &transFile,
//				          const string &emitFile
///////////////////////////////////////////////////////////////
int CNBestHMM::Initialize(const string &startFile, 
					  const string &transFile, 
					  const string &emitFile)
{

	if(!dic.Initialize(startFile, transFile, emitFile))
		return 0;
	return 1;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : NBestSearch
//	所属类名 : CNBestHMM
//	函数功能 : N Best Search算法
//	备    注 : 
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : void
//	参数说明 : int word[],观察值序列
//				 int wordNum,观察值数量
///////////////////////////////////////////////////////////////
void CNBestHMM::NBestSearch(int word[], int wordNum)
{
	/*** add by Zhu Huijia 2006-10-30 ***/
	pathNum = 0; 
	if (wordNum <= 0)
	{
		return;
	}
	/************************************/

	int wordNo = 0, stateNum = 0;		//观察值节点的总索引
	int i = 0;
	HMM_Dic::DicState *statePtr = dic.GetEmitProb(word[wordNo], stateNum);
	PathNode *headPathNode, *curNode, *lastNode;		//路径节点链表的头节点指针
	for(i=0;i<stateNum;i++)
	{
		curNode = new PathNode;
		int pos = statePtr[i].stateIndex;
		curNode->curStateIndex = pos;
		curNode->accuProb = statePtr[i].emitProb + dic.GetStartProb(pos);
		if(i == 0)
		{
			lastNode = headPathNode = curNode;
		}
		else
		{
			lastNode->next = curNode;
			lastNode = lastNode->next;
		}
	}
// 	this->headPath = headPathNode;
	PathNode *preFirst = headPathNode, *preLast = curNode , *prePtr;
			//分别代表前一个观察值的路径节点链表的头指针，尾指针，遍历指针
	PathNode *curFirst = headPathNode, *curLast = headPathNode;
			//当前观察值节点的路径链表的头指针,尾指针
	for(i = 1; i < wordNum; i++)	//观察值循环
	{
		int curPathNum = 0;
		PathNode *newHead = NULL, *newEnd = NULL;//, *newPtr = NULL;
			//分别代表当前状态的N个路径节点的头指针，尾指针，生成新节点的指针，遍历指针
			//当前状态节点下的路径数量
		statePtr = dic.GetEmitProb(word[i], stateNum);
		for(int j = 0; j < stateNum; j++)	//状态循环
		{
			for(prePtr = preFirst; prePtr != NULL; prePtr = prePtr->next) 
				//前一个观察值节点的所有路径节点的循环
			{
				double tempProb = prePtr->accuProb + statePtr[j].emitProb
							+ dic.GetTransProb(prePtr->curStateIndex, 
									statePtr[j].stateIndex);
				NewPath paths = InsertPathToTop(prePtr, newHead, newEnd, tempProb, 
								statePtr[j].stateIndex, curPathNum);
				if(paths.first != NULL)
				{
					newHead = paths.first;
					newEnd = paths.second;
				}
			}
			
			if(j == 0)
			{
				curFirst = newHead;
			}
			else
			{
				curLast->next = newHead;
				newHead->prev = curLast;
			}
			curLast = newEnd;					
			newEnd = NULL;
			newHead = NULL;
			curPathNum = 0;
		}
		preLast->next = curFirst;
		curFirst->prev = preLast;
		preFirst = curFirst;
		preLast = curLast;
	}
	SearchBack(curFirst, curLast, wordNum);
	FreeMem(headPathNode);
}

///////////////////////////////////////////////////////////////
//	函 数 名 : InsertPathToTop
//	所属类名 : CNBestHMM
//	函数功能 : 检查权值weight能否进入当前的最优的Top N中，如果能，
//				插入节点，否则返回(NULL, NULL)
//	备    注 : inline函数
//	作    者 : Truman
//	时    间 : 2003年11月4日
//	返 回 值 : NewPath (pair<PathNode*, PathNode*>) 新的Top N的首地址，尾地址
//	参数说明 : PathNode *prePtr,	此节点在这条路径之中的前一个节点指针
//             PathNode& *newFirst, TopN链表首地址
//             PathNode& *newLast,	TopN链表尾地址
//             double weight,		此节点的权重
// 			   int state,			此节点的词性
//             int &curPathNum		TopN中的节点数量（此处为引用）
///////////////////////////////////////////////////////////////
NewPath CNBestHMM::InsertPathToTop(PathNode *prePtr, 
								   PathNode *newFirst,
								   PathNode *newLast,
								   double weight,
								   int state,
								   int &curPathNum)
{
	NewPath returnPath(NULL, NULL);
	
	if(newLast != NULL && curPathNum >= MAX_N && newLast->accuProb < weight)
		return returnPath;
	//产生新的路径节点
	PathNode *newPtr = new PathNode;
	newPtr->accuProb = weight;
	newPtr->curStateIndex = state;
	newPtr->preStateIndex = prePtr->curStateIndex;
	newPtr->preNode = prePtr;
	if(curPathNum == 0)
	{
		newFirst = newPtr;
		newLast = newPtr;
		curPathNum = 1;
	}
	else
	{
		//寻找合适的位置插入
		PathNode *curPtr;
		for(curPtr = newFirst;curPtr != newLast->next; curPtr = curPtr->next)
		{
			if(newPtr->accuProb < curPtr->accuProb)
			{
				if(curPtr == newFirst)	//处理头节点情况
				{
					newFirst = newPtr;
				}
				else
				{
					curPtr->prev->next = newPtr;
					newPtr->prev = curPtr->prev;
				}
				newPtr->next = curPtr;
				curPtr->prev = newPtr;
				break;
			}
		}
		if(curPtr == newLast->next)
		{
			newLast->next = newPtr;
			newPtr->prev = newLast;
			newLast = newPtr;
		}
		if(curPathNum >= MAX_N)	//路径数是否已达到预定数量
			newLast = newLast->prev;
		else
			curPathNum++;							
	}
	PathNode *p = newLast->next;
	while(p != NULL)		//释放掉没有入选前N的路径节点
	{
		PathNode *p1 = p;
		p = p->next;
		delete p1;
	}
	newLast->next = NULL;
	returnPath.first = newFirst;
	returnPath.second = newLast;
	return returnPath;
}

void CNBestHMM::SearchBack(PathNode *pathStart, PathNode *pathEnd, int wordNum)
{
	this->pathNum = 0;		//路径数量置0
	PathNode *ptr, *newStart = NULL, *newEnd = NULL;
	NewPath paths;
	int curPathNum = 0;

	for(ptr = pathStart; ptr != NULL; ptr = ptr->next)
	{
		double temp = ptr->accuProb;
		int state = ptr->curStateIndex;
		paths = InsertPathToTop(ptr, newStart, newEnd, temp, -1, curPathNum);
		if(paths.first != NULL)
		{
			newStart = paths.first;
			newEnd = paths.second;
		}
	}

	//开始依次回退
	for(ptr = newStart; ptr != NULL; ptr = ptr->next)
	{
		PathNode *p = ptr->preNode;
		int n = wordNum;
		while(p != NULL)
		{
			this->resultPath[pathNum][--n] = p->curStateIndex;
			p = p->preNode;
		}
		this->pathWeight[pathNum] = ptr->accuProb;
		this->pathNum++;
	}
	
	/*** add by Zhu Huijia 2006-10-30 ***/
	FreeMem(newStart);
	/************************************/
}

void CNBestHMM::FreeMem(PathNode *head)
{
	PathNode *p = head;
	while(p != NULL)
	{
		PathNode *p1 = p;
		p = p->next;
		delete p1;
	}
}
