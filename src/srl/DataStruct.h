/*
 * File Name : DataStruct.h
 * Author    : hjliu
 * Time      : 2006Äê4ÔÂ4ÈÕ
 * Project   : SRLBaseline
 * Comment   : describe the data structure used in srl baseline
 *
 * Copy Right: HIT-SCIR (c) 2006-2013, all rights reserved.
 */

#ifndef _DATA_STRUCT_
#define _DATA_STRUCT_
#pragma warning(disable: 4284)

#define STL_USING_ALL
#include <stl.h>
#include <deque>

/*
 * following defines some constant string
 */

// relation
static const char *SBV = "SBV";
static const char *VOB = "VOB";
static const char *QUN = "QUN";
static const char *ADV = "ADV";

static const int SBVID = 1;
static const int VOBID = 2;
static const int QUNID = 3;
static const int ADVID = 4;

// pos
static const char *V = "v";
static const char *NT = "nt";
static const char *ND = "nd";
static const char *NL = "nl";
static const char *NS = "ns";
static const char *P = "p";
static const char *Q = "q";

// argument type
static const char *A0 = "Arg0";
static const char *A1 = "Arg1";
static const char *A0sQ = "Arg0-QTY";
static const char *A1sQ = "Arg1-QTY";
static const char *AMsTMP = "ArgM-TMP";
static const char *AMsLOC = "ArgM-LOC";
static const char *AMsDIR = "ArgM-DIR";

struct DepNode
{
    int parent;
    deque<int> dequeChildren;
    string relation;
    int id;
    pair<int, int> constituent; //the begin and end of the arg candidate
};

struct DepTree
{
    vector<DepNode> vecDepTree;
    int nodeNum;
};

struct ArgInfo
{
    int id;
    string type;
    pair<int, int> constituent;
};


#endif

