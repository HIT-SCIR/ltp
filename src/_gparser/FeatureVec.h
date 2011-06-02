#ifndef _FEATURE_VEC_
#define _FEATURE_VEC_

#pragma once
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <set>
using namespace std;

#include "MyVector.h"

/*
	class FeatureVec implements feature vector.
*/

class Feature
{
public:
	Feature(int _index=-1, double _value=0.0) : index(_index), value(_value) {}
	Feature(const Feature &other) : index(other.index), value(other.value) {}
	~Feature() {}
	Feature &operator=(const Feature &other) {
		index = other.index;
		value = other.value;
		return *this;
	}
	Feature &negation() {
		value = -value;
		return *this;
	}
	string toString() {
		ostringstream out;
		out << index << "=" << value;
		return out.str();
	}
public:
	int	index;
	double value;
};

class FeatureVec;

class SubFeatureVec {
public:
	FeatureVec *pFv;
	bool negate;
	SubFeatureVec(FeatureVec *_pFv=0, bool _negate=false) : pFv(_pFv), negate(_negate) {}
	SubFeatureVec(const SubFeatureVec &other) : pFv(other.pFv), negate(other.negate) {}
	SubFeatureVec &operator=(const SubFeatureVec &other) {
/*		if (!pFv) {
			cerr << "pFv == NULL" << endl;
		}
*/		pFv = other.pFv;
		negate = other.negate;
		return *this;
	}
};

class FeatureVec
{
private:
	vector< SubFeatureVec > subfv;
	vector<Feature> m_fv;
//	bool m_isInit;
public:
	FeatureVec() : subfv(), m_fv() {}
	FeatureVec(const vector<int> &vecKeys) : subfv(), m_fv() {
		int i = 0;
		for (; i < vecKeys.size(); ++i) {
			add(vecKeys[i], 1.0);
		}
	}
	FeatureVec(const FeatureVec &other) : m_fv(other.m_fv), subfv(other.subfv) {} 
	FeatureVec &operator=(const FeatureVec &other) {
		subfv = other.subfv;
		m_fv = other.m_fv;
		return *this;
	}

	FeatureVec &add(FeatureVec *pFv) {
		subfv.push_back( SubFeatureVec(pFv, false) ); return *this;
	}

	FeatureVec &remove(FeatureVec *pFv) {
		subfv.push_back( SubFeatureVec(pFv, true) ); return *this;
	}

	FeatureVec &add(int index, double value) {
		m_fv.push_back(Feature(index, value)); return *this;
	}

	FeatureVec &add(const Feature &feat) {
		m_fv.push_back(feat); return *this;
	}

	FeatureVec &clear() {
		m_fv.clear(); 
		subfv.clear();
		return *this;
	}

	void setKeys(const vector<int> &vecKeys) {
		subfv.clear();
		m_fv.resize(0);
		int i = 0;
		for (; i < vecKeys.size(); ++i) {
			add(vecKeys[i], 1.0);
		}
	}

	void setKeys(const MyVector<int> &vecKeys) {
		subfv.clear();
		m_fv.resize(0);
		int i = 0;
		for (; i < vecKeys.size(); ++i) {
			add(vecKeys[i], 1.0);
		}
	}

	void getKeys(vector<int> &vecKeys) const {
		vecKeys.clear();
		addKeys2List(vecKeys);
/*		set<int> setKeys;
		addKeys2Set(setKeys);
		copy(setKeys.begin(), setKeys.end(), back_inserter(vecKeys));
*/	}

	void collectFeatures() {
		map<int, double> mapFeature;
		addFeaturesToMap(mapFeature, false);
		subfv.clear();
		m_fv.clear();
		m_fv.resize(mapFeature.size());
		map<int, double>::const_iterator it = mapFeature.begin();
		int i = 0;
		for (; it != mapFeature.end() && i < m_fv.size(); ++it, ++i) {
			m_fv[i] = Feature(it->first, it->second);
		}
	}

	void addKeys2List(vector<int> &vecKeys) const;
	void addKeys2Set(set<int> &setKeys) const;

	double getScore(const vector<double> &parameters) const {
		return getScore(parameters, false);
	}

	double getScore(const vector<double> &parameters, bool negate) const;

	void addFeaturesToMap(map<int, double> &mapFv, bool negate) const;

	static double dotProduct(const FeatureVec &fv1,const FeatureVec &fv2);

	void update(vector<double> &parameters, vector<double> &total, double alpha_k, double upd) const {
		update(parameters, total, alpha_k, upd, false);
	}

	void update(vector<double> &parameters, vector<double> &total, 
		double alpha_k, double upd, bool negate) const;
};

#endif

