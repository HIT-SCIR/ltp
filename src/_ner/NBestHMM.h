/**************************************************************
	文 件 名 : NBestHMM.h
	文件功能 : 包含此模块所有数据结构的声明和常量声明
	作    者 : Truman
	创建时间 : 2003年10月25日
	项目名称 : 隐马尔可夫模型N条最优路径搜索通用算法模块
	编译环境 : Visual C++ 6.0
	备    注 : 
	历史记录 :  
**************************************************************/
#ifndef IR_NBESTHMM_H
#define IR_NBESTHMM_H

// #define STL_USING_ALL
//#include <STL.h>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include "Dictionary.h"
// #include "MemPool.h"

const int MAX_N = 1;		//N Best Search中N的最大值
const int MAX_WORDS = 2000;	//一次搜索的观察值的最大数量
using namespace std;

///////////////////////////////////////////////////////////////
//	结构体名 : PathNode
//	描    述 : 
//	历史记录 : 
//	使用说明 : 
//	作    者 : Truman
//	时    间 : 2003年10月25日
//	备    注 : 
///////////////////////////////////////////////////////////////
struct PathNode
{
	PathNode();
	~PathNode();
	double accuProb;	//路径到此节点的累计权值
	int preStateIndex;	//当前路径中前一个节点的状态值
	int curStateIndex;	//当前节点的状态值
	PathNode *preNode;	//当前路径中前一个路径节点的地址
	PathNode *next;		//指向路径节点的下一个节点
	PathNode *prev;		//路径节点总链表中的前一个节点
};

typedef pair<PathNode *, PathNode *> NewPath;
///////////////////////////////////////////////////////////////
//	类    名 : CNBestHMM
//	基    类 : 
//	描    述 : 封装HMM N Best Search算法的类
//	历史记录 : 
//	使用说明 : 
//	作    者 : Truman
//	时    间 : 2003年10月25日
//	备    注 : 
///////////////////////////////////////////////////////////////
class CNBestHMM
{
public:
	CNBestHMM(); //"DATA\\start.dat", "DATA\\trans.dat", "DATA\\emit.dat"
	virtual ~CNBestHMM();
	int Initialize(const string &startFile,
					const string &transFile,
					const string &emitFile);	//用三个文件对类进行初始化
	void NBestSearch(int word[], int wordNum);
		//word[]中存放观察值序列，wordNum为观察值序列中观察值的数量
// 	int GetResult(int **path, double weight[]);
	unsigned int GetWordsNum(void) { return dic.totalWordNum;};    

public:
	int resultPath[MAX_N][MAX_WORDS];
	double pathWeight[MAX_N];
	int pathNum;

private:
	void FreeMem(PathNode *head);
//	PathNode *headPath;	//路径链表头指针
	void SearchBack(PathNode *pathStart, PathNode *pathEnd, int wordNum);
	HMM_Dic::CDictionary dic;
	inline NewPath InsertPathToTop(PathNode *prePtr,			
							PathNode *newFirst,
							PathNode *newLast,
							double weight,
							int state,
							int &curPathNum);
};

#endif

