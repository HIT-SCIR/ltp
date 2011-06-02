#ifndef _K_BEST_PARSE_FOREST_
#define _K_BEST_PARSE_FOREST_
#pragma once
#include "ParseForestItem.h"
#include "DepInstance.h"
#include "MyLib.h"
#include "MultiArray.h"

extern const double EPS;
extern const double ZERO;
extern const double DOUBLE_NEGATIVE_INFINITY;
extern const double DOUBLE_POSITIVE_INFINITY;

#include <vector>
#include <string>
using namespace std;

/*
	this class implements parsing-chart.
*/
class ValueIndexPair;
class BinaryHeap;

class KBestParseForest
{
public:
	static int rootType;
	MultiArray<ParseForestItem> chart;

protected:
	vector<string> *sent;
	vector<string> *pos;
	int start;
	int end;
	int K;
	int tmp;

public:
	KBestParseForest() {}

	virtual KBestParseForest &reset(int _start, int _end, DepInstance *pInstance, int _K) {
		K = _K;
		start = _start;
		end = _end;
		sent = &(pInstance->forms);
		pos = &(pInstance->postags);
		vector<unsigned int> chart_dim;
		chart.setDemisionVal(chart_dim, end+1, end+1, 2, 2, K);
		chart.resize(chart_dim);
		return *this;
	}

	virtual ~KBestParseForest(void);

	bool add(int s, int type, int dir, double score, const FeatureVec &fv);

	bool add(int s, int r, int t, int type,
		int dir, int comp, double score,
		const FeatureVec &fv,
		ParseForestItem *p1, ParseForestItem *p2);

	double getProb(int s, int t, int dir, int comp) {
		return getProb(s,t,dir,comp,0);
	}

	double getProb(int s, int t, int dir, int comp, int i);

	void getProbs(int s, int t, int dir, int comp, vector<double> &vecProb); 

	void getBestParse(FeatureVec &d0, string &d1, double &parse_prob);

	void getBestParses(vector<FeatureVec> &d0, vector<string> &d1, vector<double> &parse_probs);

	void getFeatureVec(ParseForestItem &pfi, FeatureVec &fv);

	void getFeatureVec(ParseForestItem &pfi, vector<FeatureVec *> &pvfv);

	virtual void getDepString(const ParseForestItem &pfi, string &strDep);

	static string &trim(string &str) {
		remove_beg_end_spaces(str);
		return str;
	}

	void viterbi(	DepInstance *inst, 
					MultiArray<FeatureVec> &fvs, MultiArray<double>  &probs, 	
					MultiArray<FeatureVec> &nt_fvs,	MultiArray<double> &nt_probs,	
					const MultiArray<int> &static_types, bool isLabeled);

	// returns pairs of indices and -1,-1 if < K pairs
	void getKBestPairs(unsigned int chart_pos, unsigned int chart_pos2, vector< pair<int, int> > &pairs);

};

class ValueIndexPair {
public:
	double val;
	int i1, i2;
public:
	ValueIndexPair(double _val=0, int _i1=0, int _i2=0) : val(_val), i1(_i1), i2(_i2) {}

	int compareTo(const ValueIndexPair &other) const {
		if(val < other.val - EPS)
			return -1;
		if(val > other.val + EPS)
			return 1;
		return 0;
	}

	ValueIndexPair &operator=(const ValueIndexPair &other) {
		val = other.val; i1 = other.i1; i2 = other.i2; return *this;
	}
};

// Max Heap
// We know that never more than K elements on Heap
class BinaryHeap { 
private:
	int DEFAULT_CAPACITY; 
	int currentSize; 
	vector<ValueIndexPair> theArray;
public:
	bool empty() {
		return currentSize == 0;
	}
	BinaryHeap(int def_cap) {
		DEFAULT_CAPACITY = def_cap;
		theArray.resize(DEFAULT_CAPACITY+1); 
		// theArray[0] serves as dummy parent for root (who is at 1) 
		// "largest" is guaranteed to be larger than all keys in heap
		theArray[0] = ValueIndexPair(DOUBLE_POSITIVE_INFINITY,-1,-1);      
		currentSize = 0; 
	} 

	ValueIndexPair getMax() {
		return theArray[1]; 
	}

	int parent(int i) { return i / 2; } 
	int getLeftChild(int i) { return 2 * i; } 
	int getRightChild(int i) { return 2 * i + 1; } 

	void add(const ValueIndexPair &e) { 
		// bubble up: 
		int where = currentSize + 1; // new last place 
		while ( e.compareTo(theArray[parent(where)]) > 0 ){ 
			theArray[where] = theArray[parent(where)]; 
			where = parent(where); 
		} 
		theArray[where] = e; currentSize++;
	}

	void removeMax(ValueIndexPair &max);
};

#endif

