#ifndef _DEP_PIPE_
#define _DEP_PIPE_

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

#include "DepInstance.h"
#include "FeatureVec.h"
#include "Alphabet.h"
#include "CONLLReader.h"
#include "CONLLWriter.h"
#include "Parameter.h"
#include "ParserOptions.h"
#include "MultiArray.h"

class DepPipe
{
public:
	DepPipe(const ParserOptions &_options);
	~DepPipe(void);

	int initInputFile(const char *filename);
	void uninitInputFile();
	int initOutputFile(const char *filename);
	void uninitOutputFile();

	int outputInstance(const DepInstance *pInstance);

	const char *getType(int typeIndex);

	DepInstance *nextInstance();

	int createAlphabet(vector<int> &vecLength);
	void closeAlphabet();

//	int createInstances();
	int createInstances();

//	int writeInstance(DepInstance *pInstance, ofstream &featFile);
	virtual int writeInstance(FILE *featFile, DepInstance *pInstance);

public:

/*	DepInstance *readInstance(	ifstream &featFile, int length, 
								MultiArray<FeatureVec> &fvs,
								MultiArray<double> &probs,
								MultiArray<FeatureVec> &nt_fvs,
								MultiArray<double> &nt_probs,
								const Parameter &params);
*/
	virtual int readInstance(	FILE *featFile, int length, 
								MultiArray<FeatureVec> &fvs,
								MultiArray<double> &probs,
								MultiArray<FeatureVec> &fvs_trips,
								MultiArray<double> &probs_trips,
								MultiArray<FeatureVec> &fvs_sibs,
								MultiArray<double> &probs_sibs,
								MultiArray<FeatureVec> &nt_fvs,
								MultiArray<double> &nt_probs,
								FeatureVec &fv,
								string &actParseTree,
								const Parameter &params);

//	static void writeObject(ofstream &outf, const DepInstance &instance);
//	static void readObject(ifstream &inf, DepInstance &instance);

	virtual void fillFeatureVectors(DepInstance *instance,
							MultiArray<FeatureVec> &fvs,
							MultiArray<double> &probs,
							MultiArray<FeatureVec> &fvs_trips,
							MultiArray<double> &probs_trips,
							MultiArray<FeatureVec> &fvs_sibs,
							MultiArray<double> &probs_sibs,
							MultiArray<FeatureVec> &nt_fvs,		
							MultiArray<double> &nt_probs,
							const Parameter &params);

	void createSpan(DepInstance *pInstance);

	void createFeatureVector(DepInstance *pInstance);

protected:
	virtual void addExtendedFeature(DepInstance *pInstance, FeatureVec &fv) {}

	// virtual void writeExtendedFeatures(DepInstance *pInstance, ofstream &featFile) {}
	virtual void writeExtendedFeatures(DepInstance *pInstance, FILE *featFile) {}

protected:
	void mapTypes();

	void add(const string &feat, FeatureVec &fv);
	void add(const string &feat, double val, FeatureVec &fv);

	void addArcFeature(DepInstance *pInstance, int small, int large, bool attR, FeatureVec &fv);

	void addArcFeature_surrounding(const string &prefix, const vector<string> &vecVal, 
		int first, int second, const string &attDist, FeatureVec &fv);

	void addArcFeature_between(const string &prefix, const vector<string> &vecVal, 
		int first, int second, const string &attDist, FeatureVec &fv);

	void addArcFeature_unigram(DepInstance *pInstance, int nodeIdx, bool is_child, const string &dir_dist, FeatureVec &fv);
	void addArcFeature_bigram(DepInstance *pInstance, int headIdx, int childIdx, const string &dir_dist, FeatureVec &fv);

	void addArcFeature_bigram_2info(const string &prefix,
									const string &h_info1, const string &h_info2, 
									const string &c_info1, const string &c_info2, 
									const string &dir_dist, FeatureVec &fv);

	void addArcFeature_bigram_1info(const string &prefix,
									const string &h_info, const string &c_info, 
									const string &dir_dist, FeatureVec &fv);

	void addArcFeature_sur_6(const string &prefix, 
		const string &leftOf1, const string &one, const string &rightOf1,
		const string &leftOf2, const string &two, const string &rightOf2,
		const string &attDist, FeatureVec &fv);

	void addLabelFeature(DepInstance *pInstance, int nodeIdx, const string &deprel, bool is_child, bool attR, FeatureVec &fv);
//	void addLabelFeature_surrounding(const string &prefix, const vector<string> &vecInfo, int nodeIdx, const string &deprel_child, const string &dir, FeatureVec &fv);
	void addLabelFeature_surrounding(const string &prefix, const vector<string> &vecInfo, int nodeIdx, const string &deprel, const string &strIsChild, const string &dir, FeatureVec &fv);

//	void addLabeledFeature(DepInstance *pInstance, int index, const string &type,
//		bool attR, 	bool childFeatures, FeatureVec &fv);

	
//	void addTwoObsFeature(const char *prefix, const string &item1F1, const string &item1F2, 
//		const string &item2F1, const string &item2F2, const string &attachDistance,
//		FeatureVec &fv);

public:
	Alphabet m_featAlphabet;
	Alphabet m_labelAlphabet;
	DepInstance m_instance;

	vector<string> m_vecTypes;
	vector<int> m_vecTypesInt;

protected:
	const ParserOptions &options;
	DepReader *m_depReader;
	DepWriter *m_depWriter;

};

#endif
