/*
 * File Name     : MyStruct.h
 * Author        : Frumes
 *
 * Create Time   : 2006Äê12ÔÂ31ÈÕ
 * Project Name  £ºNewSRLBaseLine
 * Remark        : define some stuctures used in the project
 *
 */

#ifndef _MY_STRUCT_
#define _MY_STRUCT_
#pragma warning ( disable : 4786 )

#include <string>
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

/*----------------- typedef define begin-------------------------------------*/
typedef pair<int, int>  ArgPos;         //arg position: (begin,end)
typedef vector< ArgPos > VecPosForVerb; //the args position for current predicate
typedef vector< VecPosForVerb > VecPosForSent;   //for current sentence

typedef vector<string> VecFeatForCons;  //the all features for dependency node
typedef vector< VecFeatForCons > VecFeatForVerb; //for predicate
typedef vector< VecFeatForVerb > VecFeatForSent; //for sentence

typedef pair<string, ArgPos>  ArgInfo;  //the arg format: arg_type,arg_position

typedef map<int, vector<ArgInfo> > MapSentArg;
/*----------------- typedef define end --------------------------------------*/


/*------------- typedef define begin ----------*/
typedef struct LTPData
{
    vector<int>	   vecParent;
    vector<string> vecWord;
    vector<string> vecPos;
    vector<string> vecNe;
    vector<string> vecRelation;
} LTPData;

typedef struct DepNode
{
    int id;
    int parent;
    deque<int> dequeChildren;
    pair<int, int> constituent; //the begin and end of the arg candidate
    string relation;
} DepNode;

typedef struct DepTree
{
    int nodeNum;
    vector<DepNode> vecDepNode;
} DepTree;
/*------------- typedef define begin -----------*/


/*------------ fileName struct ----------*/
typedef struct FileNameStruct
{
    string m_strSRLConfFileName;
    string m_strSRLDicFileName;
    string m_strFeaturesFileName;
    string m_strPositionsFileName;
    string m_strPredicatesFileName;
    string m_strWordsFileName;
    string m_strDataTextFileName;
    string m_strPredictFileName;
} FileNameStruct;
/*------------ fileName struct ----------*/

/*------------ fileStream struct --------*/
typedef struct FileStreamStruct
{
    ofstream outFeaturesFile;
    ofstream outPositionsFile;
    ofstream outPredicatesFile;
    ofstream outWordsFile;
    ofstream outDataTextFile;

    ifstream inPredictFile;
} FileStreamStruct;
/*------------ fileStream struct --------*/


#endif

