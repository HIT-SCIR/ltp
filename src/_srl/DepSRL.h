// DepSRL class
// Based on Hjliu's two programs:
//   AutoCSRLIRCDPB (feature extraction portion)
//   SRLCombine (post-process and result generation potion)
// Created: 2007-09-19 By: msmouse

#ifndef _DEP_SRL_
#define _DEP_SRL_

#include"MyStruct.h"
#include"MyLib.h"
#include"SRLBaselineExt.h"
#include<vector>
#include<utility>
#include<string>

#include<maxentmodel.hpp>

class DepSRL {
public:
	DepSRL() {}

	~DepSRL() {
		if(m_resourceLoaded) {
			ReleaseResource();
		}
	}

	// Load necessary resources into memory
	int LoadResource(const string &ConfigDir = "ltp_data/srl_data/");
	// Release all resources
	int ReleaseResource();

	// Produce DepSRL result for a sentence
	int GetSRLResult(
		const vector<string> &words,
		const vector<string> &POSs,
		const vector<string> &NEs,
		const vector< pair<int, string> > &parse,
		vector< pair< int, vector< pair<string, pair< int, int > > > > > &vecSRLResult
	);

    // Produce DepSRL result for a sentence (manual predicates)
    int GetSRLResult(
        const vector<string> &words,
        const vector<string> &POSs,
        const vector<string> &NEs,
        const vector< pair<int, string> > &parse,
        const vector<int> &predicates,
        vector< pair< int, vector< pair<string, pair< int, int > > > > > &vecSRLResult
    );

    // Produce DepSRL result for a sentence (LTPData interface)
    // int DepSRL::GetSRLResult(
    int GetSRLResult(
        const LTPData     &ltpData,
        const vector<int> &predicates,
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
    );

private:
	// 1.Extract Features from input
	int ExtractFeatures(
		const LTPData              &ltpData,
		const vector<int>          &VecAllPredicates,
		VecFeatForSent &vecAllFeatures,
		VecPosForSent        &vecAllPos
	);

	// 2.Predict with the maxent library
	int Predict(
		VecFeatForSent &vecAllFeatures,
		vector< vector< pair<string, double> > > &vecAllPairMaxArgs,
		vector< vector< pair<string, double> > > &vecAllPairNextArgs
	);
	
	// 3.form the SRL result, based on predict result from maxent model
	int FormResult(
		const vector<string> &words,
		const vector<string> &POSs,
		const vector<int>    &VecAllPredicates,
		VecPosForSent  &vecAllPos,
		vector< vector< pair<string, double> > > &vecAllPairMaxArgs,
		vector< vector< pair<string, double> > > &vecAllPairNextArgs,
		vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
	);

    // 4. rename arguments to short forms (ARGXYZ->AXYZ)
    int RenameArguments(
        vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
    );



	//get parents and relations in the dependent parse tree
	void GetParAndRel(const vector< pair<int, string> >& vecParser, 
		vector<int>& vecParent, 
		vector<string>& vecRelation) const;

	//vector<const char*> to vector<string>
	//void TransVector(const vector<const char*>& vecInStr, 
	//	vector<string>& vecOutStr) const;
	
	// find verb (predicate to be tagged) in a sentence
	void GetPredicateFromSentence(const vector<string>& vecPos,
		vector<int>& vecPredicate) const;
	
	void ProcessOnePredicate(
		const vector<string>& vecWords,
		const vector<string>& vecPos,
		int intPredicates, 
		const vector< pair<int, int> >             &vecPairPS,
		const vector< pair<string, double> >       &vecPairMaxArgs, 
		const vector< pair<string, double> >       &vecPairNextArgs,
		vector< pair< string, pair< int, int > > > &vecResultForOnePredicate
	);

private:
	//////////////////////////////////////////////////////////////////////////
	//-----------------for create srl result using--------------------------//
	void FindCollisionCand(const vector< pair<int, int> >& vecPairPSCands, 
		const pair<int, int>& pairCurPSCand,
		vector<int>& vecPairColPSCands) const;
	void FindSameLabelCand(const vector< pair<string, double> >& vecPairArgCands,
		const pair<string, double>& pairCurArgCand,
		vector<int>& vecPairSameArgCands) const;
	void InsertOneArg(const pair<int, int>& pArgPS,
		const pair<string, double>& pMaxArg,
		const pair<string, double>& pNextArg,					  
		vector< pair<int, int> >& vecPairPSBuf,
		vector< pair<string, double> >& vecPairMaxArgBuf,
		vector< pair<string, double> >& vecPairNextArgBuf) const;
	void RemoveNullLabel(const vector< pair<string, double> >& vecPairMaxArgs,
		const vector< pair<string, double> >& vecPairNextArgs,
		const vector< pair<int, int> >& vecPairPS,
		vector< pair<string, double> >& vecPairNNLMax,
		vector< pair<string, double> >& vecPairNNLNext,
		vector< pair<int, int> >& vecPairNNLPS) const;
	bool IsInsertNColLabel(const vector<int>& vecCol,
		const pair<string, double>& pArgCand,
		vector< pair<string, double> >& vecPairMaxArgBuf,
		vector< pair<string, double> >& vecPairNextArgBuf,
		vector< pair<int, int> >& vecPairPSBuf) const;
	bool IsInsertSameLabel(const vector<int>& vecSame,
		const pair<string, double>& pArgCand,
		vector< pair<string, double> >& vecPairMaxArgBuf,
		vector< pair<string, double> >& vecPairNextArgBuf,
		vector< pair<int, int> >& vecPairPSBuf,
        vector<int> &vctSameDel) const;
	bool IsInsertPredicate(int intPredicate,
		vector< pair<string, double> >& vecPairMaxArgBuf,
		vector< pair<int, int> >& vecPairPSBuf) const;
	//-----------------for create srl result using--------------------------//
	//////////////////////////////////////////////////////////////////////////

private:
	//////////////////////////////////////////////////////////////////////////
	//-------------------------for post process-----------------------------//
	void PostProcess(const vector<string>& vecPos,
		const vector< pair<int, int> >& vecPairPS,	
		const vector< pair<string, double> >& vecPairMaxArgs,
		const vector< pair<string, double> >& vecPairNextArgs,					 
		vector< pair<int, int> >& vecPairPSBuf,
		vector< pair<string, double> >& vecPairMaxArgsBuf,
		vector< pair<string, double> >& vecPairNextArgsBuf) const;
	void QTYArgsProcess(const vector<string>& vecPos,
		vector< pair<int, int> >& vecPairPSBuf,
		vector< pair<string, double> >& vecPairMaxArgsBuf,
		vector< pair<string, double> >& vecPairNextArgsBuf) const;
	void PSERArgsProcess(const string& strArgPrefix,
		const vector<string>& vecPos,
		const vector< pair<int, int> >& vecPairPS,
		const vector< pair<string, double> >& vecPairMaxArgs,
		const vector< pair<string, double> >& vecPairNextArgs,
		vector< pair<int, int> >& vecPairPSBuf,
		vector< pair<string, double> >& vecPairMaxArgsBuf,
		vector< pair<string, double> >& vecPairNextArgsBuf) const;
	bool FindArgFromDropCand(const string& strArgPat,
		const vector< pair<int, int> >& vecPairPS,
		const vector< pair<string, double> >& vecPairMaxArgs,
		const vector< pair<string, double> >& vecPairNextArgs,
		pair<int, int>& pArgPS,
		pair<string, double>& pMaxArg,
		pair<string, double>& pNextArg) const;
	void ReplaceArgFromNextProp(const vector<int>& vecIndex,
		vector< pair<int, int> >& vecPairPSBuf,
		vector< pair<string, double> >& vecPairMaxArgsBuf,
		vector< pair<string, double> >& vecPairNextArgsBuf) const;
	bool IsPosPattern(int intBegin,
		int intEnd,
		const vector<string>& vecPos,
		const string& strPattern) const;
	bool IsMaxPropGreaterThreshold(double dThreSholdVal,
		const vector<int>& vecIndex, 
		const vector< pair<string, double> >& vecPairMaxArgsBuf) const;
	//-------------------------for post process-----------------------------//
	//////////////////////////////////////////////////////////////////////////



private:
	bool                m_resourceLoaded;
	SRLBaselineExt*		m_srlBaseline;

	maxent::MaxentModel *m_maxentModel;
};

#endif
