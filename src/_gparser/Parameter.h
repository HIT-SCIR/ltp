#ifndef _PARAMETER_
#define _PARAMETER_

#pragma once
#include <vector>
#include <string>
using namespace std;

#include "DepInstance.h"
#include "FeatureVec.h"
#include "MyVector.h"
#include "ParserOptions.h"

extern const double EPS;
extern const double ZERO;
extern const double DOUBLE_NEGATIVE_INFINITY;


class Parameter
{
public:
	Parameter(int size, const ParserOptions &_options) : options(_options) {
		m_parameters.resize(0);
		m_parameters.resize(size, 0.0);
		m_total.resize(0);
		m_total.resize(size, 0.0);
		m_lossType = "punc";
		m_SCORE = 0.0;
	}

	~Parameter(void) {}

	void setLoss(const string &it) {
		m_lossType = it;
	}

	void setParams(const MyVector<double> &parameters) {
		m_parameters.resize(parameters.size(), 0.0);
		copy(parameters.begin(), parameters.begin()+parameters.size(), m_parameters.begin());
		m_total.resize(0);
		m_total.resize(parameters.size(), 0.0);
		m_lossType = "punc";
		m_SCORE = 0.0;
	}

	void averageParams(double avVal) {
		int j = 0;
		for (; j < m_total.size(); ++j) {
//			m_total[j] *= 1.0/avVal;
			m_parameters[j] = m_total[j] / avVal;
		}
	}

	void storeParamsToTmp() {
		m_vecTmpParameters.resize(m_parameters.size());
		copy(m_parameters.begin(), m_parameters.end(), m_vecTmpParameters.begin());
	}

	void restoreParamsFromTmp() {
		m_parameters.resize(m_vecTmpParameters.size());
		copy(m_vecTmpParameters.begin(), m_vecTmpParameters.end(), m_parameters.begin());
	}

	void updateParamsMIRA(DepInstance *pInstance, vector<FeatureVec> &d0, vector<string> &d1, double upd);

	double getScore(const FeatureVec &fv) const{
		return fv.getScore(m_parameters);
	}

	void hildreth(const vector<FeatureVec> &a, const vector<double> &b, vector<double> &alpha, int K);

	double numErrors(DepInstance *pInstance, const string &pred, const string &act)
	{
		if(m_lossType == "nopunc") {
			if (options.m_isLabeled) {
				return numErrorsLabelNoPunc(pInstance, pred, act);
			} else {
				return numErrorsArcNoPunc(pInstance, pred, act);
			}
		} else {
			if (options.m_isLabeled) {
				return numErrorsLabel(pInstance, pred, act);
			} else {
				return numErrorsArc(pInstance, pred, act);
			}
		}
	}

	double numErrorsArc(DepInstance *pInstance, const string &pred, const string &act);

	double numErrorsLabel(DepInstance *pInstance, const string &pred, const string &act);

	double numErrorsArcNoPunc(DepInstance *pInstance, const string &pred, const string &act);

	bool isPunctuation(const string &str) {
		return false;
	}

	double numErrorsLabelNoPunc(DepInstance *pInstance, const string &pred, const string &act);
	void show() {
		int interval = m_parameters.size() / 50;
		cerr << endl;
		int i = 0;
		for (; i < m_parameters.size(); i += interval) {
			cerr << "(" << i << " " << m_parameters[i] << " " << m_total[i] << ")\t";
			if (i % 10 >= 0) cerr << endl;
		}
/*		cerr << endl;
		int i = 1;
		for (i; i < m_parameters.size() && i <= 10; ++i) {
			cerr << "(" << i << " " << m_parameters[i] << " " << m_total[i] << ")\t";
			if (i % 10 >= 0) cerr << endl;
		}
		cerr << endl << endl;
		int len = m_parameters.size();
		for (i=1; len - i >= 0 && i <= 10; ++i) {
			cerr << "(" << len-i << " " << m_parameters[len-i] << " " << m_total[len-i] << ")\t";
			if (i % 10 >= 0) cerr << endl;
		}
		cerr << endl;
*/	}
public:
	vector<double> m_parameters;
	vector<double> m_total;
	string m_lossType;
	double m_SCORE;

	vector<double> m_vecTmpParameters;

	const ParserOptions &options;
};

#endif

