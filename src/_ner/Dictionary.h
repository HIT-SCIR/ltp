/**************************************************************
	文 件 名 : Dictionary.h
	文件功能 : CDictionary类的声明文件
	作    者 : Truman
	创建时间 : 2003年10月27日
	项目名称 : NBestHMM
	编译环境 : Visual C++ 6.0
	备    注 : 
	历史记录 :  
**************************************************************/

#ifndef IR_CDICTIONARY_H
#define IR_CDICTIONARY_H

// #define STL_USING_ALL
// #include <STL.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

const int MAX_STATE_NUM = 120;	//HMM中状态数量最大值

using namespace std;

namespace HMM_Dic
{

///////////////////////////////////////////////////////////////
//	结构体名 : DicState
//	描    述 : 保存一个状态和其相应的发射频率
//	历史记录 : 
//	使用说明 : 
//	作    者 : Truman
//	时    间 : 2003年10月27日
//	备    注 : 
///////////////////////////////////////////////////////////////
struct DicState
{
	int stateIndex;			//状态序号
	double emitProb;		//在stateIndex状态下的发射频率
};

///////////////////////////////////////////////////////////////
//	结构体名 : DicWord
//	描    述 : 保存观察值序号和此观察值对应的状态数量，状态数组
//	历史记录 : 
//	使用说明 : 
//	作    者 : Truman
//	时    间 : 2003年10月27日
//	备    注 : stateArray指向的是一个new出来的数组，在CDictionary的
//				析构函数中进行delete
///////////////////////////////////////////////////////////////
struct DicWord
{
	int wordIndex;		//观察值序号
	int stateNum;		//此观察值节点的状态数量
	DicState *stateArray;	//状态节点数组的头指针
};

class CDictionary  
{
public:
	long GetFileLength(const string &fileName);
	CDictionary();
	virtual ~CDictionary();
	bool Initialize(const string &startFile,
					const string &transFile,
					const string &emitFile);	//用三个文件对类进行初始化
	DicState *GetEmitProb(int wordIndex, int &returnStateNum) {
		returnStateNum = wordArray[wordIndex].stateNum;
		return wordArray[wordIndex].stateArray;
	};	
		//返回wordIndex词的状态数组的首地址，状态数量放在returnStateNum中返回
	double GetTransProb(int stateIndex1, int stateIndex2) {
		return transProb[stateIndex1][stateIndex2];
	};
		//取stateIndex1状态到stateIndex2的转移频率
	double GetStartProb(int stateIndex) {
		return startProb[stateIndex];
	};
		//取stateIndex的初始频率
	void DestoryData(void);
	int totalWordNum;	//字典里观察值的总数量
private:
	DicWord *wordArray;		//观察值数组的头指针
	int totalStateNum;	//此HMM模型中的状态数量
	double transProb[MAX_STATE_NUM][MAX_STATE_NUM];
						//转移概率矩阵
	double startProb[MAX_STATE_NUM];
						//初始概率数组
	bool ReadEmitFile(const string &emitFileName);
	bool ReadTransFile(const string &transFileName);
	bool ReadStartFile(const string &startFileName);
};
}

#endif
