#ifndef _DEP_INSTANCE_
#define _DEP_INSTANCE_

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

#include "FeatureVec.h"
#include "MyLib.h"

/*
	this class implements the representation of parsing result of one sentence.

*/
class DepInstance
{
public:
	DepInstance() {}
	~DepInstance() {}
	int size() { return forms.size(); }
	void resize(int _size) { forms.resize(_size); }

/*	void writeObject(ofstream &outf) const;
	void readObject(ifstream &inf);
	void setInstance(const vector<string> &_forms,
					 const vector<string> &_lemmas,
					 const vector<string> &_cpostags,
					 const vector<string> &_postags,
					 const vector< vector<string )
*/

public:
	FeatureVec fv;
	string actParseTree;
	vector<string> forms;
	vector<string> lemmas;
	vector<string> cpostags;
	vector<string> postags;
	vector< vector<string> > feats;
	vector<int> heads;
	vector<string> deprels;

	vector< vector<string> > k_deprels;
	vector< vector<int> > k_heads;
	vector<double> k_probs;
};

#endif

