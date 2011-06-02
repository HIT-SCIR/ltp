///////////////////////////////////////////////////////////////
//	File Name     :MyStruct.h
//	File Function :
//	Author 	      : Frumes
//	Create Time   : 2006年12月31日
//	Project Name  ：NewSRLBaseLine
//	Operate System : 
//	Remark        : define some stuctures used in the project
//	History：     : 
///////////////////////////////////////////////////////////////

#ifndef _MY_STRUCT_
#define _MY_STRUCT_
#pragma warning ( disable : 4786 )

//#include <stdio.h>
//#include <string.h>
#include <string>
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

//----------------- typedef define begin--------------------------------------//
typedef pair<int, int>  ArgPos; //arg position: (begin,end)
typedef vector< ArgPos > VecPosForVerb; //the args position for current predicate
typedef vector< VecPosForVerb > VecPosForSent; //for current sentence

typedef vector<string> VecFeatForCons; //the all features for dependency node
typedef vector< VecFeatForCons > VecFeatForVerb; //for predicate
typedef vector< VecFeatForVerb > VecFeatForSent; //for sentence

typedef pair<string, ArgPos>  ArgInfo; //the arg format: arg_type,arg_position

typedef map<int, vector<ArgInfo> > MapSentArg;
//----------------- typedef define end --------------------------------------//


//----------------- typedef define begin --------------------------------------//
///////////////////////////////////////////////////////////////
//	Struct Name : LTPData
//	Description : 
//	Function    : 
//	History	    : 
//	Instruction : 
//	Author	    : Frumes
//	Time	    : 2006年12月31日
//	Remark	    : 
///////////////////////////////////////////////////////////////
typedef struct LTPData
{
	vector<int>	   vecParent;
	vector<string> vecWord;
	vector<string> vecPos;		
	vector<string> vecNe;	
	vector<string> vecRelation;
} LTPData;


///////////////////////////////////////////////////////////////
//	Struct Name : DepNode
//	Description : 
//	Function    : 
//	History	    : 
//	Instruction : 
//	Author	    : Frumes
//	Time	    : 2006年12月31日
//	Remark	    : 
///////////////////////////////////////////////////////////////
typedef struct DepNode
{
    int id;
	int parent;
    deque<int> dequeChildren;    
    pair<int, int> constituent; //the begin and end of the arg candidate
	string relation;
} DepNode;


///////////////////////////////////////////////////////////////
//	Struct Name : DepTree
//	Description : 
//	Function    : 
//	History	    : 
//	Instruction : 
//	Author	    : Frumes
//	Time	    : 2006年12月31日
//	Remark	    : 
///////////////////////////////////////////////////////////////
typedef struct DepTree
{
	int nodeNum;
	vector<DepNode> vecDepNode;	
} DepTree;
//----------------- typedef define begin --------------------------------------//	


//------------ fileName struct ----------------------------------//
typedef struct FileNameStruct
{	
	string	 m_strSRLConfFileName;
	string	 m_strSRLDicFileName;
	string   m_strFeaturesFileName;
	string	 m_strPositionsFileName;
	string	 m_strPredicatesFileName;
	string	 m_strWordsFileName;
	string   m_strDataTextFileName;
	string	 m_strPredictFileName;
} FileNameStruct;
//------------ fileName struct ----------------------------------//

//------------ fileStream struct ----------------------------------//
typedef struct FileStreamStruct
{
	ofstream outFeaturesFile;
	ofstream outPositionsFile;
	ofstream outPredicatesFile;
	ofstream outWordsFile;
	ofstream outDataTextFile;

	ifstream inPredictFile;
} FileStreamStruct;
//------------ fileStream struct ----------------------------------//


#endif
