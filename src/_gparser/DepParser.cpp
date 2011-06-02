#include "DepParser.h"
#include <cstdio>
#include <sstream>
using namespace std;

void DepParser::train(const vector<int> &instanceLengths) 
{
	const set<int> &setIterNums = options.m_setTrain_IterNums_to_SaveParamModel;
	int i = 1;
	for(; i <= options.m_numIter; i++) {
		cerr << "Iteration " << i << endl;
		trainingIter(instanceLengths, i);

		if (!setIterNums.empty() && setIterNums.end() != setIterNums.find(i)) {
			cerr << "\n++++++++++++\n";
			cerr << "save parameter model at iter: " << i << endl;
			print_time();
			params.storeParamsToTmp();
			params.averageParams(i * instanceLengths.size());
			ostringstream out;
			out << i;
			saveParamModel(options.m_strModelName.c_str(), out.str().c_str());
			params.restoreParamsFromTmp();
			print_time();
			cerr << "\n++++++++++++\n";
		}
	}

	params.averageParams(options.m_numIter * instanceLengths.size());
}

void DepParser::trainingIter(const vector<int> &instanceLengths, int iter)
{
	int numUpd = 0;
	FILE *trainForest = 0;
	
	if (!options.m_strTrainForestFile.empty()) {
		trainForest = fopen(options.m_strTrainForestFile.c_str(), "rb");
		if (!trainForest) {
			cerr << "open train forest err: " << options.m_strTrainForestFile << endl;
			return;
		}
	} else {
		if (0 > pipe.initInputFile(options.m_strTrainFile.c_str())) return;
	}

	int numInstances = instanceLengths.size();
	DepInstance local_inst;
	DepInstance *pInst = 0;

	int i = 0;
	for(; i < numInstances; i++) {
		if((i) % options.m_display_interval == 0) cerr<< i << " ";

		int length = instanceLengths[i];
		allocMultiArr(length);
		if (!options.m_strTrainForestFile.empty()) {
			pInst = &local_inst;
			pInst->resize(length);

			if (0 > pipe.readInstance(trainForest,length,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs, pInst->fv, pInst->actParseTree, params)) break;
		} else {
			pInst = pipe.nextInstance();
			if (!pInst) break;
			pipe.createFeatureVector(pInst);
			pipe.createSpan(pInst);
			pipe.fillFeatureVectors(pInst,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,params);
		}

		double upd = (double)(options.m_numIter*numInstances - (numInstances*(iter-1)+(i+1)) + 1);
		int K = options.m_trainK;
		vector<FeatureVec> d0;
		vector<string> d1;
		vector<double> parse_probs;

		decoder.decodeProjective(pInst,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,K, d0, d1, parse_probs);			

/*		int k = 0; 
		for (; k < d1.size(); ++k) {
			cerr << endl << params.getScore(d0[k]) << endl;
			cerr << d1[k] << endl;
		}
*/
//		cerr << "\nupd: " << upd << endl;
		params.updateParamsMIRA(pInst, d0, d1, upd);
//		params.show();
	}

	cerr << "\ninstance num: " << numInstances << endl;

	if (!options.m_strTrainForestFile.empty() && trainForest) {
		fclose(trainForest);
	} else {
		pipe.uninitInputFile();
	}
}

//////////////////////////////////////////////////////
// Get Best Parses ///////////////////////////////////
//////////////////////////////////////////////////////
void DepParser::outputParses ()
{
	if (0 > pipe.initInputFile(options.m_strTestFile.c_str())) return;
	if (0 > pipe.initOutputFile(options.m_strOutFile.c_str())) return;

	cerr << "Processing Sentence: ";
	DepInstance *instance = pipe.nextInstance();
	int cnt = 0;
	while(instance) {
		if(++cnt % options.m_display_interval == 0) cerr<< cnt << " ";

		int length = instance->forms.size();
		allocMultiArr(length);

		pipe.fillFeatureVectors(instance,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,params);

		int K = 1; //default
		if (options.m_testK > 0) {
			K = options.m_testK;
		}

		vector<FeatureVec> d0;
		vector<string> d1;
		vector<double> parse_probs;
		decoder.decodeProjective(instance,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,K, d0, d1, parse_probs);			

		if (parse_probs.empty() || parse_probs[0] < DOUBLE_NEGATIVE_INFINITY + EPS) {
			cerr << "instance: " << cnt << " parse err: returned 0 result" << endl;
			exit(0);
		}

		instance->k_heads.clear();
		instance->k_deprels.clear();
		instance->k_probs.clear();

		if (options.m_testK > 0) {
			fillInstance_k(*instance, d1, parse_probs);
		} else {
			fillInstance(*instance, d1[0], parse_probs[0]);
		}


		pipe.outputInstance(instance);

		if ( options.m_numMaxInstance > 0 && cnt == options.m_numMaxInstance) break;
		instance = pipe.nextInstance();
	}

	pipe.uninitInputFile();
	pipe.uninitOutputFile();

	cerr << "\ninstance num: " << cnt << endl;
}

int DepParser::initInstance(DepInstance &inst,
				 const vector<string> &vecWord,
				 const vector<string> &vecCPOS)
{
	if (vecWord.empty() || vecWord.size() != vecCPOS.size()) {
		cerr << "gparser param error: word, CPOS num not equal!" << endl;
		return -1;
	}

	inst.forms.resize(vecWord.size()+1);
	inst.lemmas.resize(vecWord.size()+1);
	inst.cpostags.resize(vecWord.size()+1);
	inst.postags.resize(vecWord.size()+1);
	inst.feats.resize(vecWord.size()+1);
	inst.heads.resize(vecWord.size()+1);
	inst.deprels.resize(vecWord.size()+1);

	inst.forms[0] = "<root>";
	inst.lemmas[0] = "<root-LEMMA>";
	inst.cpostags[0] = "<root-CPOS>";
	inst.deprels[0] = "<no-type>";
	inst.heads[0] = -1;

	copy(vecWord.begin(), vecWord.end(), inst.forms.begin()+1);
	copy(vecCPOS.begin(), vecCPOS.end(), inst.cpostags.begin()+1);

	return 0;
}
int DepParser::getParseResult(const DepInstance &inst,
				   vector<int> &vecHead,
				   vector<string> &vecRel)
{
	if (inst.heads.size() != inst.deprels.size()) {
		cerr << "gparser parse err: heads and deprels num not equal." << endl;
		return -1;
	}
	vecHead.resize(inst.heads.size()-1);
	vecRel.resize(inst.deprels.size()-1);
	copy(inst.heads.begin()+1, inst.heads.end(), vecHead.begin());
	copy(inst.deprels.begin()+1, inst.deprels.end(), vecRel.begin());
	return 0;
}

int DepParser::parseSent(const vector<string> &vecWord,
			  const vector<string> &vecPOS,
			  vector<int> &vecHead,
			  vector<string> &vecRel)
{
	DepInstance inst;
	if (0 != initInstance(inst, vecWord, vecPOS)) return -1;
	DepInstance *pInst = &inst;

	int length = pInst->forms.size();
	allocMultiArr(length);

	pipe.fillFeatureVectors(pInst, fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,params);

	int K = 1; //default

	vector<FeatureVec> d0;
	vector<string> d1;
	vector<double> parse_probs;
	decoder.decodeProjective(pInst,fvs,probs,fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs,nt_probs,K, d0, d1, parse_probs);			

	if (parse_probs.empty() || parse_probs[0] < DOUBLE_NEGATIVE_INFINITY + EPS) {
		cerr << " parse err: returned 0 result" << endl;
		return -1;
	}

	fillInstance(*pInst, d1[0], parse_probs[0]);

	if (0 != getParseResult(inst, vecHead, vecRel)) return -1;
	if (vecHead.size() != vecWord.size()) {
		cerr << "gparser parse err: word and head num not equal." << endl;
		return -1;
	}

	return 0;
}

void DepParser::fillParseResult( const string &tree_span, double prob, vector<int> &heads, vector<string> &deprels )
{
	vector<string> triples;
	split_bychar(tree_span, triples, ' ');

	int node_num = triples.size();

//	if (node_num + 1 != 
	heads.resize(node_num+1);		// heads[0] is not used.
	deprels.resize(node_num+1);

	int j = 1;
	for(; j < heads.size(); ++j) {
		int triple_idx = j - 1;
		vector<string> head_child_rel;
		split_bychars(triples[triple_idx], head_child_rel, "|:");

		if (head_child_rel.size() != 3) {
			cerr << "tree span format err: " << triples[triple_idx] << endl;
			cerr << "whole span: [" << tree_span << "]" << endl;
			deprels[j] = "ERR";
			heads[j] = -1;
			continue;
		}

		const string &strDepRelIdx = head_child_rel[2];
		const string &strHead = head_child_rel[0];

		if (options.m_isLabeled) {
			int typeIdx = atoi(strDepRelIdx.c_str());
			deprels[j] = pipe.getType(typeIdx);
			if (deprels[j].empty()) {
				cerr << "deprel err: idx = " << strDepRelIdx << endl;
			}
		}
		
		if (j == 1 && options.m_isOutPutScore) { // 将概率输出到第一个词的dep_relation后面
			ostringstream out;
			out << prob;
			deprels[j] += "###" + out.str();
		}
		heads[j] =atoi(strHead.c_str());
	}
}

void DepParser::fillInstance_k(DepInstance &inst, const vector<string> d1, const vector<double> &parse_probs)
{
	int i = 0;
	for (; i < parse_probs.size(); ++i) {
		if (parse_probs[i] < DOUBLE_NEGATIVE_INFINITY + EPS) {
			cerr << "parse err: return only " << i << " results." << endl;
			break;
		}

		inst.k_probs.push_back(parse_probs[i]);
		inst.k_heads.push_back(vector<int>());
		inst.k_deprels.push_back(vector<string>());
		fillParseResult(d1[i], inst.k_probs.back(), inst.k_heads.back(), inst.k_deprels.back());
	}
}

void DepParser::fillInstance( DepInstance &inst, const string &tree_span, double prob )
{
	fillParseResult(tree_span, prob, inst.heads, inst.deprels);
}

int DepParser::saveParamModel(const char *modelName, const char *paramModelIterNum)
{
	string strFileName = "parameter.";
	strFileName += paramModelIterNum;
	strFileName += ".";
	strFileName += modelName;

	cerr << "save parameter model: " << strFileName << endl;
	print_time();

	FILE *paramModel = fopen(strFileName.c_str(), "wb");
	if (!paramModel) {
		cerr << "DepParser::saveModel() open file err: " << strFileName << endl;
		return -1;
	}
	writeObject(paramModel, params.m_parameters);
	fclose(paramModel);
	cerr << "done!" << endl;
	print_time();

	return 0;
}

int DepParser::loadParamModel(const char *modelPath, const char *modelName, const char *paramModelIterNum)
{
	string strFileName = modelPath;
	strFileName += "parameter.";
	strFileName += paramModelIterNum;
	strFileName += ".";
	strFileName += modelName;

	cerr << "load parameter model: " << strFileName << endl;
	print_time();

	FILE *paramModel = fopen(strFileName.c_str(), "rb");
	if (!paramModel) {
		cerr << "DepParser::loadParamModel() open file err: " << strFileName << endl;
		return -1;
	}

	MyVector<double> parameters;
	readObject(paramModel, parameters);
	params.setParams(parameters);

	fclose(paramModel);
	cerr << "done!" << endl;
	print_time();
	return 0;
}


int DepParser::saveAlphabetModel(const char *modelName)
{
	string strFileName = "alphabet.";
	strFileName += modelName;
	cerr << "save alphabet model: " << strFileName << endl;
	print_time();

	ofstream alphabetModel(strFileName.c_str());
	if (!alphabetModel) {
		cerr << "DepParser::saveAlphabetModel() open file err: " << strFileName << endl;
		return -1;
	}
	pipe.m_featAlphabet.writeObject(alphabetModel);
	writeObject(alphabetModel, int(-2));
	pipe.m_labelAlphabet.writeObject(alphabetModel);
	writeObject(alphabetModel, int(-3));
	alphabetModel.close();
	cerr << "done!" << endl;
	print_time();
	return 0;
}


int DepParser::loadAlphabetModel(const char *modelPath, const char *modelName)
{
	string strFileName = modelPath;
	strFileName += "alphabet.";
	strFileName += modelName;
	cerr << "load alphabet model: " << strFileName << endl;
	print_time();

	ifstream alphabetModel(strFileName.c_str());
	if (!alphabetModel) {
		cerr << "DepParser::loadAlphabetModel() open file err: " << strFileName << endl;
		return -1;
	}
	int tag;
	pipe.m_featAlphabet.readObject(alphabetModel);
	readObject(alphabetModel, tag);
	if (tag != -2) {
		cerr << "DepParser::loadAlphabetModel() err, not see -2" << endl;
		return -1;
	}
	pipe.m_labelAlphabet.readObject(alphabetModel);
	readObject(alphabetModel, tag);
	if (tag != -3) {
		cerr << "DepParser::loadAlphabetModel() err, not see -3" << endl;
		return -1;
	}

	alphabetModel.close();
	cerr << "done!" << endl;
	print_time();
	return 0;
}

