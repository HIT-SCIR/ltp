/**************************************************************
	文 件 名 : Dictionary.cpp
	文件功能 : CDictionary类的实现文件
	作    者 : Truman
	创建时间 : 2003年10月27日
	项目名称 : NBestHMM
	编译环境 : Visual C++ 6.0
	备    注 : 
	历史记录 :  
**************************************************************/
//#include "StdAfx.h"
#include <stdlib.h>
#include "Dictionary.h"

namespace HMM_Dic
{

CDictionary::CDictionary()
{
	totalWordNum = 0;
}

CDictionary::~CDictionary()
{
	DestoryData();
}

///////////////////////////////////////////////////////////////
//	函 数 名 : Initialize
//	所属类名 : CDictionary
//	函数功能 : 对类进行初始化，包括读取三个文件
//	备    注 : 
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : bool
//	参数说明 : const string &startFile,
//				         const string &transFile,
//				         const string &emitFile
///////////////////////////////////////////////////////////////
bool CDictionary::Initialize(const string &startFile,
							const string &transFile,
							const string &emitFile)
{
	if(!ReadStartFile(startFile))
		return false;
	if(!ReadTransFile(transFile))
		return false;
	if(!ReadEmitFile(emitFile))
		return false;
	return true;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : ReadEmitFile
//	所属类名 : CDictionary
//	函数功能 : 读取发射概率文件
//	备    注 : 如果读取失败,则返回false
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : bool
//	参数说明 : const string &emitFileName
///////////////////////////////////////////////////////////////
bool CDictionary::ReadEmitFile(const string &fileName)
{
	long fileSize = GetFileLength(fileName);
	if(fileSize == -1)
		return false;
	char *buf = new char[fileSize];
	FILE *fp;
	fp = fopen(fileName.c_str(), "rb");
	if(fp == NULL)
		return false;
	fread(buf, fileSize, 1, fp);

	int wordIndex = 0;	//词的索引
	int posIndex = 0;
	int prev = 0, pos = 0;
	char tempBuf[20];	//存放临时字符串（待处理字符串）
	tempBuf[0] = '\0';
	int tempIndex = 0;	//临时字符串的索引，指向最后一个字符
	int posNum;
	int status = 0;		//状态（当前读入的是什么值）
				//status = 0,	读入的是观察值数量
				//status = 1,	读入的是状态数量
				//status = 2,	读入的是状态号
				//status = 3,	读入的是发射概率
	for(long i = 0; i < fileSize; i++)
	{
		if(buf[i] == 10)	//回车换行
		{
			if(tempIndex == 0)		//空行
			{
				continue;
			}
			tempBuf[tempIndex] = '\0';
			tempIndex = 0;
			switch(status)
			{
			case 0:
				this->totalWordNum = atoi(tempBuf);
				this->wordArray = new DicWord[totalWordNum];
				status = 1;
				break;
			case 3:
				wordArray[wordIndex].stateArray[posIndex].emitProb = atof(tempBuf);
				posIndex = 0;
				status = 1;
				wordIndex++;
				if(wordIndex >= this->totalWordNum)
					return true;
				break;
			}
			continue;
		}
		if(buf[i] == 32)		//空格
		{
			if(tempIndex == 0)		//空行
				continue;
			tempBuf[tempIndex] = '\0';
			tempIndex = 0;
			switch(status) {
			case 1:
				posNum = atoi(tempBuf);
				wordArray[wordIndex].stateNum = posNum;
				wordArray[wordIndex].stateArray = new DicState[posNum];
				status = 2;
				break;
			case 2:
				wordArray[wordIndex].stateArray[posIndex].stateIndex = atoi(tempBuf);
				status = 3;
				break;
			case 3:
				wordArray[wordIndex].stateArray[posIndex].emitProb = atof(tempBuf);
				posIndex++;
				status = 2;
				break;
			default:
				break;
			}
			continue;
		}
		
		tempBuf[tempIndex] = buf[i];
		tempIndex++;
	}
	delete [] buf;
	return true;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : ReadStartFile
//	所属类名 : CDictionary
//	函数功能 : 读取初始概率文件
//	备    注 : 读取文件失败返回false
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : bool
//	参数说明 : const string &startFileName
///////////////////////////////////////////////////////////////
bool CDictionary::ReadStartFile(const string &startFileName)
{
	ifstream startFile;
	startFile.open(startFileName.c_str());
	if(!startFile)		//Read file error!
		return false;
	int n;
	startFile >> n;
	totalStateNum = n;
	for(int i=0;i<n;i++)
	{
		startFile >> startProb[i];
	}
	startFile.close();
	return true;
}

///////////////////////////////////////////////////////////////
//	函 数 名 : ReadTransFile
//	所属类名 : CDictionary
//	函数功能 : 读取转移概率文件
//	备    注 : 读取文件失败返回false
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : bool
//	参数说明 : const string &transFileName
///////////////////////////////////////////////////////////////
bool CDictionary::ReadTransFile(const string &transFileName)
{
	ifstream transFile;
	transFile.open(transFileName.c_str());
	if(!transFile)		//Read file error!
		return false;
	for(int i=0;i<totalStateNum;i++)
		for(int j=0;j<totalStateNum;j++)
			transFile >> transProb[i][j] ;
	
	transFile.close();
	return true;
}


///////////////////////////////////////////////////////////////
//	函 数 名 : DestoryData
//	所属类名 : CDictionary
//	函数功能 : 销毁CDictionary类中的数据，以便读取新的数据
//	备    注 : 
//	作    者 : Truman
//	时    间 : 2003年10月28日
//	返 回 值 : void
//	参数说明 : 
///////////////////////////////////////////////////////////////
void CDictionary::DestoryData()
{
	if(totalWordNum != 0)
	{
		for(int i=0;i<totalWordNum;++i)
			delete [] wordArray[i].stateArray;
		delete [] wordArray;
		totalWordNum = 0;
	}
}

///////////////////////////////////////////////////////////////
//	函 数 名 : GetFileLength
//	所属类名 : CDictionary
//	函数功能 : 读文件的长度
//	备    注 : 
//	作    者 : Truman
//	时    间 : 2003年11月8日
//	返 回 值 : long	文件长度(字节数),如果读取错误则返回-1
//	参数说明 : const string &fileName	文件名
///////////////////////////////////////////////////////////////
long CDictionary::GetFileLength(const string &fileName)
{
	if (fileName.empty()) 
	   return -1; 
	FILE *fp = fopen(fileName.c_str(), "rb"); 
	if (fp == NULL) 
	   return -1; 
	fseek(fp, 0 , SEEK_END); 
	long lResult = ftell(fp); 
	fclose(fp); 
	return lResult; 
}

}


