#include "DepPipe.h"
#include "MyVector.h"
#include <iterator>
using namespace std;

DepPipe::DepPipe(const ParserOptions &_options) : options(_options)
{
	m_depReader = 0;
	m_depReader = new CONLLReader();
	m_depWriter = new CONLLWriter();
}

DepPipe::~DepPipe(void)
{
	if (m_depReader) delete m_depReader;
	if (m_depWriter) delete m_depWriter;
}
int DepPipe::initInputFile(const char *filename) {
	if (0 != m_depReader->startReading(filename)) return -1;
	return 0;
}

void DepPipe::uninitInputFile() {
	if (m_depWriter) m_depReader->finishReading();
}

int DepPipe::initOutputFile(const char *filename) {
	if (0 != m_depWriter->startWriting(filename)) return -1;
	return 0;
}

void DepPipe::uninitOutputFile() {
	if (m_depWriter) m_depWriter->finishWriting();
}

int DepPipe::outputInstance(const DepInstance *pInstance) {
	if (0 != m_depWriter->write(pInstance)) return -1;
	return 0;
}


const char *DepPipe::getType(int typeIndex) {
	if (typeIndex >= 0 && typeIndex < m_vecTypes.size()) {
		return m_vecTypes[typeIndex].c_str();
	} else {
		return "";
	}
}

int DepPipe::createAlphabet(vector<int> &instanceLength) 
{
	cerr << "Creating Alphabet..." << endl;

	initInputFile(options.m_strTrainFile.c_str());
	m_featAlphabet.allowGrowth();
	m_labelAlphabet.allowGrowth();

	instanceLength.clear();

	DepInstance *pInstance = nextInstance();
	int numInstance = 0;

	while (pInstance) {
		if (++numInstance % options.m_display_interval == 0) cerr << numInstance << " ";

		createSpan(pInstance);
		createFeatureVector(pInstance);

		instanceLength.push_back(pInstance->size());

//		const vector<string> &deprels = pInstance->deprels;
//		int i = 0;
//		for (; i < deprels.size(); ++i) {
//			int id = m_labelAlphabet.lookupIndex(deprels[i]);
//			cerr << deprels[i] << " " << id << endl;
//			exit(0);
//		}
//		m_labelAlphabet.show();
//		exit(0);

		if ( options.m_numMaxInstance > 0 && numInstance == options.m_numMaxInstance) break;
		pInstance = nextInstance();
	}
	
	uninitInputFile();

	cerr << endl;
	cerr << "instance num: " << numInstance << endl;
	cerr << "Features num: " <<  m_featAlphabet.size() << endl;
//	m_featAlphabet.show();
//	m_labelAlphabet.show();
	cerr << "label num: " <<  m_labelAlphabet.size() << endl;
	cerr << "Create Alphabet Done" << endl;
	return 0;
}


void DepPipe::closeAlphabet() {
	m_featAlphabet.stopGrowth();
	m_labelAlphabet.stopGrowth();
	mapTypes();
}

void DepPipe::mapTypes() {
	m_vecTypes.resize(m_labelAlphabet.size());
	vector<string> vecKeys;
	m_labelAlphabet.getKeys(vecKeys);
	int i = 0;
	for(; i < vecKeys.size(); ++i) {
		int idx = m_labelAlphabet.lookupIndex(vecKeys[i]);
		if (idx < 0 || idx >= m_labelAlphabet.size()) {
			cerr << "m_labelAlphabet err: " << vecKeys[i] << " : " << idx << endl;
			continue;
		}
		m_vecTypes[idx] = vecKeys[i];
	}
}

int DepPipe::createInstances()
{
	cerr << "Create Instances: " << endl;

	if (0 > initInputFile(options.m_strTrainFile.c_str())) return -1;

	FILE *featFile = fopen(options.m_strTrainForestFile.c_str(), "wb");
	if (!featFile) {
		cerr << "open train forest file err: " << options.m_strTrainForestFile << endl;
		return -1;
	}
	DepInstance *pInstance = nextInstance();

	int numInstance = 0;

	while (pInstance) {
		if (++numInstance % options.m_display_interval == 0) cerr << numInstance << " ";

		createFeatureVector(pInstance);
		createSpan(pInstance);
//		cerr << pInstance->actParseTree << endl;

		writeInstance(featFile, pInstance);
		if ( options.m_numMaxInstance > 0 && numInstance == options.m_numMaxInstance) break;
		pInstance = nextInstance();
	}

	cerr << "instance num: " << numInstance << endl;

	uninitInputFile();
	fclose(featFile);
	return 0;
}

DepInstance *DepPipe::nextInstance() 
{
	DepInstance *pInstance = m_depReader->getNext();
	if (pInstance && pInstance->forms.empty()) return 0;
	return pInstance;
}

void DepPipe::createSpan(DepInstance *pInstance) {
	const vector<string> &deprels = pInstance->deprels;
	const vector<int> &heads = pInstance->heads;
	string &spans = pInstance->actParseTree;
	spans = "";
	m_labelAlphabet.lookupIndex(deprels[0]);
	int i = 1;
	for (; i < deprels.size(); ++i) {
		ostringstream out;
		out << heads[i] << "|" << i << ":" << m_labelAlphabet.lookupIndex(deprels[i]);
		spans += out.str();
		if (i < deprels.size()-1) spans += " ";
	}
//	cerr << spans << endl;
}

void DepPipe::add(const string &feat, FeatureVec &fv) {
	int num = m_featAlphabet.lookupIndex(feat);
	if (num >= 0) {
		fv.add(num, 1.0);
	}
}
void DepPipe::add(const string &feat, double val, FeatureVec &fv) {
	int num = m_featAlphabet.lookupIndex(feat);
	if (num >= 0) {
		fv.add(num, val);
	}
}

void DepPipe::addArcFeature(DepInstance *pInstance, int small, int large, bool attR, FeatureVec &fv) 
{

	const vector<string> &forms = pInstance->forms;
	const vector<string> &lemmas = pInstance->lemmas;
	const vector<string> &cpostags = pInstance->cpostags;
	const vector<string> &postags = pInstance->postags;

	string att = attR ? "_R" : "_L";

	int dist = abs(large - small);
	string strDist;
	if (dist > 10) {
		strDist = "10";
	} else if (dist > 5) {
		strDist = "5";
	} else {
		ostringstream out;
		out << dist-1;
		strDist = out.str();
	}
	string attDist = att + "_" + strDist;

/*
	string feattmp = "1" + forms[small] + "_" + forms[large];
	add(feattmp , fv);	
	add(feattmp + att, fv);
	add(feattmp + attDist, fv);
	feattmp = "2" + forms[small] + "_" + cpostags[large];
	add(feattmp , fv);	
	add(feattmp + att, fv);
	add(feattmp + attDist, fv);
	feattmp = "3" + cpostags[small] + "_" + forms[large];
	add(feattmp , fv);	
	add(feattmp + att, fv);
	add(feattmp + attDist, fv);
	feattmp = "4" + cpostags[small] + "_" + cpostags[large];
	add(feattmp , fv);	
	add(feattmp + att, fv);
	add(feattmp + attDist, fv);
	return;
//*/
	if (options.m_isUseCPostag) {
		addArcFeature_between("cpos", cpostags, small, large, attDist, fv);
		addArcFeature_surrounding("cpos", cpostags, small, large, attDist, fv);

	}
	if (options.m_isUsePostag) {
		addArcFeature_between("pos", postags, small, large, attDist, fv);
		addArcFeature_surrounding("pos", postags, small, large, attDist, fv);
	}

	int headIndex = attR ? small : large;
	int childIndex = !attR ? small : large;
	
	addArcFeature_unigram(pInstance, headIndex, false, attDist, fv);
	addArcFeature_unigram(pInstance, childIndex, true, attDist, fv);
	addArcFeature_bigram(pInstance, headIndex, childIndex, attDist, fv);
}

void DepPipe::addArcFeature_unigram(DepInstance *pInstance, int nodeIdx, bool is_child, const string &dir_dist, FeatureVec &fv)
{
	string strIsChild = is_child ? "_1" : "_0";

	const string &form = pInstance->forms[nodeIdx];
	const string &lemma = pInstance->lemmas[nodeIdx];
	const string &postag = pInstance->postags[nodeIdx];
	const string &cpostag = pInstance->cpostags[nodeIdx];

	string prefix = "uni"; prefix += "2";
	string feat;
	if (options.m_isUseForm && options.m_isUsePostag) {
		feat = prefix + "1" + form + " " + postag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
	if (options.m_isUseLemma && options.m_isUsePostag) {
		feat = prefix + "2" + lemma + " " + postag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
	if (options.m_isUseForm && options.m_isUseCPostag) {
		feat = prefix + "3" + form + " " + cpostag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
	if (options.m_isUseLemma && options.m_isUseCPostag) {
		feat = prefix + "4" + lemma + " " + cpostag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}

	prefix = "uni"; prefix += "1";

	if (options.m_isUseForm) {
		feat = prefix + "1" + form + strIsChild;
		feat += dir_dist; add(feat, fv);
		add(feat, fv);
	}	

	if (options.m_isUseLemma) {
		feat = prefix + "2" + lemma + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
	if (options.m_isUsePostag) {
		feat = prefix + "3" + postag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
	if (options.m_isUseCPostag) {
		feat = prefix + "4" + cpostag + strIsChild;
		add(feat, fv);
		feat += dir_dist; add(feat, fv);
	}
}


void DepPipe::addArcFeature_bigram(DepInstance *pInstance, int headIdx, int childIdx, const string &dir_dist, FeatureVec &fv)
{
	const string &h_form = pInstance->forms[headIdx];
	const string &h_lemma = pInstance->lemmas[headIdx];
	const string &h_postag = pInstance->postags[headIdx];
	const string &h_cpostag = pInstance->cpostags[headIdx];
	const string &c_form = pInstance->forms[childIdx];
	const string &c_lemma = pInstance->lemmas[childIdx];
	const string &c_postag = pInstance->postags[childIdx];
	const string &c_cpostag = pInstance->cpostags[childIdx];

	string prefix = "bi"; prefix += "2";

	if (options.m_isUseForm && options.m_isUsePostag) {
		string prefix_2info = prefix + "1";
		addArcFeature_bigram_2info(prefix_2info, h_form, h_postag, c_form, c_postag, dir_dist, fv);
	}
	if (options.m_isUseLemma && options.m_isUsePostag) {
		string prefix_2info = prefix + "2";
		addArcFeature_bigram_2info(prefix_2info, h_lemma, h_postag, c_lemma, c_postag, dir_dist, fv);
	}
	if (options.m_isUseForm && options.m_isUseCPostag) {
		string prefix_2info = prefix + "3";
		addArcFeature_bigram_2info(prefix_2info, h_form, h_cpostag, c_form, c_cpostag, dir_dist, fv);
	}
	if (options.m_isUseLemma && options.m_isUseCPostag) {
		string prefix_2info = prefix + "4";
		addArcFeature_bigram_2info(prefix_2info, h_lemma, h_cpostag, c_lemma, c_cpostag, dir_dist, fv);
	}

	prefix = "bi"; prefix += "1";
	string prefix_1info;

	if (options.m_isUseForm) {
		prefix_1info = prefix + "1";
		addArcFeature_bigram_1info(prefix_1info, h_form, c_form, dir_dist, fv);
	}	
	if (options.m_isUseLemma) {
		string prefix_1info = prefix + "2";
		addArcFeature_bigram_1info(prefix_1info, h_lemma, c_lemma, dir_dist, fv);
	}
	if (options.m_isUsePostag) {
		string prefix_1info = prefix + "3";
		addArcFeature_bigram_1info(prefix_1info, h_postag, c_postag, dir_dist, fv);
	}
	if (options.m_isUseCPostag) {
		string prefix_1info = prefix + "4";
		addArcFeature_bigram_1info(prefix_1info, h_cpostag, c_cpostag, dir_dist, fv);
	}
}

void DepPipe::addArcFeature_bigram_2info(const string &prefix,
								const string &h_info1, const string &h_info2, 
								const string &c_info1, const string &c_info2, 
								const string &dir_dist, FeatureVec &fv)
{
	string feat;

	feat = prefix + "1" + h_info1 + " " + h_info2 + " " + c_info1 + " " + c_info2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix + "2" + h_info2 + " " + c_info1 + " " + c_info2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix + "2" + h_info1 + " " + c_info1 + " " + c_info2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix + "4" + h_info1 + " " + h_info2 + " " + c_info2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix + "5" + h_info1 + " " + h_info2 + " " + c_info1;
	add(feat, fv);	
	add(feat + dir_dist, fv);

	// h1_c2; h2_c1
	feat = prefix + "6" + h_info1 + " " + c_info2;
	add(feat, fv);	
	add(feat + dir_dist, fv);

	feat = prefix + "7" + h_info2 + " " + c_info1;
	add(feat, fv);	
	add(feat + dir_dist, fv);
}

void DepPipe::addArcFeature_bigram_1info(const string &prefix,
								const string &h_info, const string &c_info, 
								const string &dir_dist, FeatureVec &fv)
{
	string feat = prefix + h_info + " " + c_info;
	add(feat, fv);
	add(feat + dir_dist, fv);
}


void DepPipe::addArcFeature_surrounding(const string &prefix, const vector<string> &vecVal, 
							   int first, int second, const string &attDist, FeatureVec &fv) 
{
	string firstLeft = first > 0 ? vecVal[first-1] : "BEG"; // l-pos-1
	string firstRight = first < second-1 ? vecVal[first+1] : "MID"; // l-pos+1
	string secondLeft = second > first+1 ? vecVal[second-1] : "MID"; // r-pos-1
	string secondRight = second < vecVal.size()-1 ? vecVal[second+1] : "END"; // r-pos+1

	string prefix2 = prefix + "sur";
	addArcFeature_sur_6(prefix2, firstLeft, vecVal[first], firstRight,
						secondLeft, vecVal[second], secondRight, attDist, fv);
}

void DepPipe::addArcFeature_between(const string &prefix, const vector<string> &vecVal, 
										int first, int second, const string &attDist, FeatureVec &fv) 
{
	string feat_prefix = prefix + "bet" + vecVal[first] + " " + vecVal[second];
	if (options.m_isUse_arc_bet_each) {
		int i = first + 1;
		for (; i < second; ++i) {
			string feat = feat_prefix + " " + vecVal[i];
			add(feat, fv);
			add(feat + attDist, fv);
		}
	}

	if (options.m_isUse_arc_bet_same_num) {
		int l_same_ctr = 0;
		int r_same_ctr = 0;
		int i = first + 1;
		for (; i < second; ++i) {
			if (vecVal[i] == vecVal[first]) ++l_same_ctr;
			if (vecVal[i] == vecVal[second]) ++r_same_ctr;
		}
		// l_r_l-same-num
		ostringstream l_out;
		l_out << l_same_ctr;
		string l_feat = feat_prefix + "_l" + l_out.str();
		add(l_feat, fv);
		add(l_feat + attDist, fv);

		// l_r_r-same-num
		ostringstream r_out;
		r_out << r_same_ctr;
		string r_feat = feat_prefix + "_r" + r_out.str();
		add(r_feat, fv);
		add(r_feat + attDist, fv);
	}
}

void DepPipe::addArcFeature_sur_6(const string &prefix, 
								const string &leftOf1, const string &one, const string &rightOf1,
								const string &leftOf2, const string &two, const string &rightOf2,
								const string &dir_dist, FeatureVec &fv)
{
	string prefix1 = prefix + "4";
	string prefix2 = one + " " + two;
	string feat;

	// 4 elements
	// l   r   l-1 l+1 # l   r   l-1  r-1  #  l   r   l-1 r+1
	// l   r   l+1 r-1 # l   r   l+1  r+1  #  l   r   r-1 r+1	
	feat = prefix1 + "1" + prefix2 + " " + leftOf1 + " " + rightOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "2" + prefix2 + " " + leftOf1 + " " + leftOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "3" + prefix2 + " " + leftOf1 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "4" + prefix2 + " " + rightOf1 + " " + leftOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "5" + prefix2 + " " + rightOf1 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "6" + prefix2 + " " + leftOf2 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);
	
	// 3 elements
	prefix1 = prefix + "3";
	prefix2 = one + " " + two;

	feat = prefix1 + "1" + prefix2 + " " + leftOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "2" + prefix2 + " " + rightOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "3" + prefix2 + " " + leftOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "4" + prefix2 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	// 3-elements without l (one) or r (two)
	// l   l-1 r-1 # l   l-1 r+1 # l   l+1 r-1 # l   l+1 r+1
	prefix1 = prefix + "L3";

	feat = prefix1 + "1" + one + " " + leftOf1 + " " + leftOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "2" + one + " " + leftOf1 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "3" + one + " " + rightOf1 + " " + leftOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "4" + one + " " + rightOf1 + " " + rightOf2;
	add(feat, fv);
	add(feat + dir_dist, fv);

	// r   r-1 l-1 # r   r-1 l+1 # r   r+1 l-1 # r   r+1 l+1
	prefix1 = prefix + "R3";

	feat = prefix1 + "1" + two + " " + leftOf2 + " " + leftOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "2" + two + " " + leftOf2 + " " + rightOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "3" + two + " " + rightOf2 + " " + leftOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);

	feat = prefix1 + "4" + two + " " + rightOf2 + " " + rightOf1;
	add(feat, fv);
	add(feat + dir_dist, fv);
}

void DepPipe::addLabelFeature(DepInstance *pInstance, int nodeIdx, const string &deprel, bool is_child, bool attR, FeatureVec &fv) 
{
	string dir = attR ? "_R" : "_L";
	string strIsChild = is_child ? "_1" : "_0";
	string dir_child = dir + strIsChild;

	const string &form = pInstance->forms[nodeIdx];
	const string &lemma = pInstance->lemmas[nodeIdx];
	const string &postag = pInstance->postags[nodeIdx];
	const string &cpostag = pInstance->cpostags[nodeIdx];

/*
	string att = dir_child;
	string feattmp = string("lbl") + "1" + form + "_" + deprel;
	add(feattmp , fv);	
	add(feattmp + att, fv);
	feattmp = string("lbl") + "2" + cpostag + "_" + deprel;
	add(feattmp , fv);	
	add(feattmp + att, fv);	
	return;
//*/

	string feat = "lbl";
	feat += deprel + dir;
	add(feat, fv);

	string prefix = "lbl2"; prefix += deprel;

	if (options.m_isUseForm && options.m_isUsePostag && options.m_isUseForm_label) {
		feat = prefix + "1" + form + " " + postag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}
	if (options.m_isUseLemma && options.m_isUsePostag && options.m_isUseLemma_label) {
		feat = prefix + "2" + lemma + " " + postag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}
	if (options.m_isUseForm && options.m_isUseCPostag && options.m_isUseForm_label) {
		feat = prefix + "3" + form + " " + cpostag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}
	if (options.m_isUseLemma && options.m_isUseCPostag && options.m_isUseLemma_label) {
		feat = prefix + "4" + lemma + " " + cpostag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}

	prefix = "lbl1"; prefix += deprel;

	if (options.m_isUseForm && options.m_isUseForm_label) {
		feat = prefix + "1" + form;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}

	if (options.m_isUseLemma && options.m_isUseLemma_label) {
		feat = prefix + "2" + lemma;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}
	if (options.m_isUsePostag) {
		feat = prefix + "3" + postag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}
	if (options.m_isUseCPostag) {
		feat = prefix + "4" + cpostag;
		if (options.m_isUse_label_feats_t) add(feat, fv);
		if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
		add(feat + dir_child, fv);
	}

	// surrounding features
	if (options.m_isUsePostag) {
		prefix = "lbl-sur-pos";
		addLabelFeature_surrounding(prefix, pInstance->postags, nodeIdx, deprel, strIsChild, dir, fv);
	}
	if (options.m_isUseCPostag) {
		prefix = "lbl-sur-cpos";
		addLabelFeature_surrounding(prefix, pInstance->cpostags, nodeIdx, deprel, strIsChild, dir, fv);
	}
}

void DepPipe::addLabelFeature_surrounding(const string &prefix, const vector<string> &vecInfo, int nodeIdx, const string &deprel, const string &strIsChild, const string &dir, FeatureVec &fv)
{
	string left = nodeIdx > 0 ? vecInfo[nodeIdx-1] : "BEG"; // i-1
	string right = nodeIdx < vecInfo.size()-1 ? vecInfo[nodeIdx+1] : "END"; //i+1
	string dir_child = dir + strIsChild;
	string feat;
	// i-1 i i+1  # i-1 i  # i i+1
	feat = prefix + "_"  + deprel +  "1" + left + " " + vecInfo[nodeIdx] + " " + right;
	if (options.m_isUse_label_feats_t) add(feat, fv);
	if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
	add(feat + dir_child, fv);

	feat = prefix + "_"  + deprel + "2" + left + " " + vecInfo[nodeIdx];
	if (options.m_isUse_label_feats_t) add(feat, fv);
	if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
	add(feat + dir_child, fv);

	feat = prefix + "_"  + deprel +  "3" + vecInfo[nodeIdx] + " " + right;
	if (options.m_isUse_label_feats_t) add(feat, fv);
	if (options.m_isUse_label_feats_t_child) add(feat + strIsChild, fv);
	add(feat + dir_child, fv);
}

void DepPipe::createFeatureVector(DepInstance *pInstance) 
{
	pInstance->fv.clear();
	const vector<string> &deprels = pInstance->deprels;
	const vector<int> &heads = pInstance->heads;
	FeatureVec &fv = pInstance->fv;
	int i = 0;
	int length = pInstance->forms.size();
	for (; i < length; ++i) {
		if (heads[i] == -1) continue;
		int small = i < heads[i] ? i : heads[i];
		int large = i > heads[i] ? i : heads[i];
		bool attR = i < heads[i] ? false : true;
		addArcFeature(pInstance, small, large, attR, fv);
		if (options.m_isLabeled) {
			addLabelFeature(pInstance, i, deprels[i], true, attR, fv);
			addLabelFeature(pInstance, heads[i], deprels[i], false, attR, fv);
		}
	}
	addExtendedFeature(pInstance, fv);
}

void DepPipe::fillFeatureVectors(DepInstance *instance,
								 MultiArray<FeatureVec> &fvs,
								 MultiArray<double> &probs,
								 MultiArray<FeatureVec> &fvs_trips,
								 MultiArray<double> &probs_trips,
								 MultiArray<FeatureVec> &fvs_sibs,
								 MultiArray<double> &probs_sibs,
								 MultiArray<FeatureVec> &nt_fvs,		
								 MultiArray<double> &nt_probs,
								 const Parameter &params)
{
	int instanceLength = instance->size();
	vector<unsigned int> fvs_dim(3);
	unsigned int fvs_pos;

	// Get production crap.	
	int w1 = 0;
	for(; w1 < instanceLength; w1++) {
		int w2 = w1+1;
		if (w2 >= instanceLength) continue;
		fvs.setDemisionVal(fvs_dim, w1, w2, 0);
		fvs.getElement(fvs_dim, fvs_pos);
		for(; w2 < instanceLength; w2++) {
			int ph = 0;
			for(; ph < 2; ph++) {
				bool attR = ph == 0 ? true : false;
				addArcFeature(instance,w1,w2,attR, fvs.getElement(fvs_pos));
				probs.getElement(fvs_pos) = params.getScore(fvs.getElement(fvs_pos));
				vector<int> vecKeys;
				fvs.getElement(fvs_pos).getKeys(vecKeys);
				// cerr << vecKeys.size() << "\t";
				++fvs_pos;
			}
		}
		//		cerr << endl;
	}

	if(options.m_isLabeled) {
		vector<unsigned int> nt_dim(4);
		unsigned int nt_pos;
		nt_fvs.setDemisionVal(nt_dim, 0, 0, 0, 0);
		nt_fvs.getElement(nt_dim, nt_pos);
		int w1 = 0;
		for(; w1 < instanceLength; w1++) {
			int t = 0;
			for(; t < m_vecTypes.size(); t++) {
				const string &type = m_vecTypes[t];
				int ph = 0;
				for(; ph < 2; ph++) {
					bool attR = ph == 0 ? true : false;
					int ch = 0;
					for(; ch < 2; ch++) {
						bool child = ch == 0 ? true : false;
						addLabelFeature(instance, w1, type, child, attR, nt_fvs.getElement(nt_pos));
						nt_probs.getElement(nt_pos) = params.getScore(nt_fvs.getElement(nt_pos));
						vector<int> vecKeys;
						nt_fvs.getElement(nt_pos).getKeys(vecKeys);
						// cerr << vecKeys.size() << "\t";
						++nt_pos;						
					}
				}
			}
			//			cerr << endl;
		}
	}		
}


int DepPipe::writeInstance(FILE *featFile, DepInstance *pInstance)
{
	//	cerr << endl;
	int instanceLength = pInstance->size();
	for(int w1 = 0; w1 < instanceLength; w1++) {
		for(int w2 = w1+1; w2 < instanceLength; w2++) {
			for(int ph = 0; ph < 2; ph++) {
				bool attR = ph == 0 ? true : false;
				FeatureVec prodFV;
				addArcFeature(pInstance,w1,w2,attR,prodFV);
				vector<int> vecKeys;
				prodFV.getKeys(vecKeys);
				// cerr << vecKeys.size() << " ";
				::writeObject(featFile, vecKeys);
			}
		}
		// cerr << endl;
	}
	::writeObject(featFile, (int)-1);

	if(options.m_isLabeled) {
		for(int w1 = 0; w1 < instanceLength; w1++) {	    
			for(int t = 0; t < m_vecTypes.size(); t++) {
				const string &type = m_vecTypes[t];	
				for(int ph = 0; ph < 2; ph++) {
					bool attR = ph == 0 ? true : false;
					for(int ch = 0; ch < 2; ch++) {
						bool child = ch == 0 ? true : false;
						FeatureVec prodFV;
						addLabelFeature(pInstance, w1, type, child, attR, prodFV);
						vector<int> vecKeys;
						prodFV.getKeys(vecKeys);
						// cerr << vecKeys.size() << " ";
						// copy(vecKeys.begin(), vecKeys.end(), ostream_iterator<int>(cerr, " "));
						// cerr << endl;
						::writeObject(featFile, vecKeys);
					}
				}
			}
			// cerr << endl;
		}
		::writeObject(featFile, int(-2));
	}
	//	exit(0);
	writeExtendedFeatures(pInstance,featFile);

	vector<int> vecKeys;
	pInstance->fv.getKeys(vecKeys);
	::writeObject(featFile, vecKeys);
	::writeObject(featFile, int(-3));

	//	cerr << pInstance->actParseTree.size() << endl;
	writeObject(featFile, pInstance->actParseTree);


	writeObject(featFile, int(-4));
	return 0;
}


int DepPipe::readInstance(FILE *featFile, int length, 
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
						  const Parameter &params)
{
	//	cerr << "read instance" << endl;
	//	cerr << endl;
	vector<unsigned int> fvs_dim;
	unsigned int fvs_pos;

	MyVector<int> vecKeys;

	// Get production crap.
	for(int w1 = 0; w1 < length; w1++) {
		int w2 = w1 + 1;
		if (w2 >= length) continue;
		fvs.setDemisionVal(fvs_dim, w1, w2, 0);
		fvs.getElement(fvs_dim, fvs_pos);
		for(; w2 < length; w2++) {
			for(int ph = 0; ph < 2; ph++) {
				FeatureVec &prodFV = fvs.getElement(fvs_pos);
				::readObject(featFile, vecKeys);
				//				cerr << vecKeys.size() << " ";
				prodFV.setKeys(vecKeys);
				probs.getElement(fvs_pos) = params.getScore(prodFV);
				++fvs_pos;
			}
		}
		//		cerr << endl;
	}
	int last;
	::readObject(featFile, last);
	if(last != -1) { cerr << "DepPipe::readInstance() Error reading file. -1" << endl; return -1; }

	//	cerr << endl;

	if(options.m_isLabeled) {
		vector<unsigned int> nt_dim(4);
		unsigned int nt_pos;
		nt_fvs.setDemisionVal(nt_dim, 0, 0, 0, 0);
		nt_fvs.getElement(nt_dim, nt_pos);
		for(int w1 = 0; w1 < length; w1++) {
			for(int t = 0; t < m_vecTypes.size(); t++) {
				const string &type = m_vecTypes[t];
				for(int ph = 0; ph < 2; ph++) {						
					for(int ch = 0; ch < 2; ch++) {
						FeatureVec &prodFV = nt_fvs.getElement(nt_pos);
						::readObject(featFile, vecKeys);
						prodFV.setKeys(vecKeys);
						//						cerr << vecKeys.size() << " ";
						nt_probs.getElement(nt_pos) = params.getScore(prodFV);
						++nt_pos;
					}
				}
			}
			//			cerr << endl;
		}
		::readObject(featFile, last);
		if(last != -2) { cerr << "DepPipe::readInstance() Error reading file.-2 vs. " << last << endl; return -1; }
	}

	//	exit(0);

	::readObject(featFile, vecKeys);
	fv.setKeys(vecKeys);
	::readObject(featFile, last);
	if(last != -3) { cerr << "DepPipe::readInstance() Error reading file.-3 vs. " << last << endl; return -1; }

	MyVector<char> my_str;
	::readObject(featFile, my_str);
	actParseTree = my_str.begin();
	//	cerr << actParseTree << endl;
	//	cerr << actParseTree.size() << endl;
	::readObject(featFile, last);
	if(last != -4) { cerr << "DepPipe::readInstance() Error reading file.-4 vs. " << last << endl; return -1; }
	return 0;
}

/*
void DepPipe::addLabeledFeature(DepInstance *pInstance, int nodeIdx, const string &type,
								bool dir, 	bool is_child, FeatureVec &fv) 
{
	const vector<string> &forms = pInstance->forms;
	const vector<string> &lemmas = pInstance->lemmas;
	const vector<string> &cpostags = pInstance->cpostags;
	const vector<string> &postags = pInstance->postags;

	if(!options.m_isLabeled) {
		cerr << "DepPipe::addLabeledFeature(): not labeled, should not call this function" << endl;
		return; 
	}

	const vector<string> &forms = pInstance->forms;
	const vector<string> &cpostags = pInstance->cpostags;
	string dir_child = dir ? "R" : "L";
	string strIsChild = is_child ? "1" : "0";
	dir_child = "_" + dir_child + "_" + strIsChild;

	string form = forms[nodeIdx];
	string cpostag = cpostags[nodeIdx];

	string left = nodeIdx > 0 ? cpostags[nodeIdx-1] : "BEG";	// pos-1
	string right = nodeIdx < cpostags.size()-1 ? cpostags[nodeIdx+1] : "END"; // pos+1

	/*
		t				t_dir_is-child	[BUG!]
		f_t 			*
		p_t 			*
		f_p_t 			*
		p-1_p_t 		*
		p_p+1_t 		*
		p-1_p_p+1 		*


	string prefix = "lbl";
	string feat = prefix + "1" + type;
	add(feat, fv);
	add(feat + dir_child, fv);			// ???

	if (!options.m_isUseForm) {
		return;
	}

	feat = prefix + "2" + form + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);

	if (!options.m_isUseCPostag) {
		return;
	}

	feat = prefix + "3" + cpostag + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);

	feat = prefix + "4" + form + " " + cpostag + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);

	feat = prefix + "5" + left + " " + cpostag + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);

	feat = prefix + "6" + cpostag + " " + right + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);

	feat = prefix + "7" + left + " " + cpostag + right + " " + " " + type;
	add(feat, fv);
	add(feat + dir_child, fv);
}


/**
* Add features for two items, each with two observations, e.g. head,
* head pos, child, and child pos.
*
* The use of StringBuilders is not yet as efficient as it could
* be, but this is a start. (And it abstracts the logic so we can
* add other features more easily based on other items and
* observations.)
**/
/*
void DepPipe::addTwoObsFeature(const char *_prefix, const string &item1F1, const string &item1F2, 
							   const string &item2F1, const string &item2F2, const string &attachDistance,
							   FeatureVec &fv) 
{	// bi-gram features
	string prefix = _prefix;
	string feat = prefix + "2FF1=" + item1F1;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF1=" + item1F1 + " " + item1F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF1=" + item1F1 + " " + item1F2 + " " + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF1=" + item1F1 + " " + item1F2 + " " + item2F2 + " " + item2F1;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF2=" + item1F1 + " " + item2F1;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF3=" + item1F1 + " " + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);


	feat = prefix + "2FF4=" + item1F2 + " " + item2F1;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF4=" + item1F2 + " " + item2F1 + " " + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF5=" + item1F2 + " " + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF6=" + item2F1 + " " + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF7=" + item1F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF8=" + item2F1;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);

	feat = prefix + "2FF9=" + item2F2;
	add(feat, fv);
	feat += "*" + attachDistance;
	add(feat, fv);
}

*/

/*
int DepPipe::createInstances()
{
if (0 > initInputFile(options.m_strTrainFile.c_str())) return -1;
ofstream featFile(options.m_strTrainForestFile.c_str());
if (!featFile) {
cerr << "open feat file err: " << options.m_strTrainForestFile << endl;
return -1;
}
DepInstance *pInstance = nextInstance();

int numInstance = 0;
cerr << "Create Feature Vector Instances: " << endl;

while (pInstance) {
if (++numInstance % options.m_display_interval == 0) cerr << numInstance << " ";

createFeatureVector(pInstance);
createSpan(pInstance);

writeInstance(pInstance, featFile);

if ( options.m_numMaxInstance > 0 && numInstance == options.m_numMaxInstance) break;
pInstance = nextInstance();
}

cerr << "instance num: " << numInstance << endl;
//	exit(0);
uninitInputFile();
featFile.close();
return 0;
}
*/
/*
int DepPipe::writeInstance(DepInstance *pInstance, ofstream &featFile)
{
int instanceLength = pInstance->size();
for(int w1 = 0; w1 < instanceLength; w1++) {
for(int w2 = w1+1; w2 < instanceLength; w2++) {
for(int ph = 0; ph < 2; ph++) {
bool attR = ph == 0 ? true : false;
FeatureVec prodFV;
addArcFeature(pInstance,w1,w2,attR,prodFV);
vector<int> vecKeys;
prodFV.getKeys(vecKeys);
//cerr << vecKeys.size() << " ";
::writeObject(featFile, vecKeys);
}
}
}
::writeObject(featFile, (int)-3);

if(m_isLabeled) {
for(int w1 = 0; w1 < instanceLength; w1++) {	    
for(int t = 0; t < m_vecTypes.size(); t++) {
const string &type = m_vecTypes[t];	
for(int ph = 0; ph < 2; ph++) {
bool attR = ph == 0 ? true : false;
for(int ch = 0; ch < 2; ch++) {
bool child = ch == 0 ? true : false;
FeatureVec prodFV;
addLabeledFeature(pInstance,w1,
type, attR,child,prodFV);
vector<int> vecKeys;
prodFV.getKeys(vecKeys);
//cerr << vecKeys.size() << " ";
::writeObject(featFile, vecKeys);
}
}
}
}
::writeObject(featFile, int(-3));
}

writeExtendedFeatures(pInstance,featFile);

vector<int> vecKeys;
pInstance->fv.getKeys(vecKeys);
::writeObject(featFile, vecKeys);
::writeObject(featFile, int(-4));

writeObject(featFile, *pInstance);
::writeObject(featFile, int(-1));
return 0;
}

*/

/*
DepInstance *DepPipe::readInstance(ifstream &featFile, int length, 
MultiArray<FeatureVec> &fvs,
MultiArray<double> &probs,
MultiArray<FeatureVec> &nt_fvs,
MultiArray<double> &nt_probs,
const Parameter &params)
{
vector<unsigned int> fvs_dim(3);
unsigned int fvs_pos;
fvs.setDemisionVal(fvs_dim, length, length, 2);
probs.resize(fvs_dim);
if (0 > fvs.resize(fvs_dim)) return 0;	// Get production crap.
for(int w1 = 0; w1 < length; w1++) {
int w2 = w1 + 1;
if (w2 >= length) continue;
fvs.setDemisionVal(fvs_dim, w1, w2, 0);
fvs.getElement(fvs_dim, fvs_pos);
for(; w2 < length; w2++) {
for(int ph = 0; ph < 2; ph++) {
FeatureVec &prodFV = fvs.getElement(fvs_pos);
vector<int> vecKeys;
::readObject(featFile, vecKeys);
//cerr << vecKeys.size() << " ";
prodFV.setKeys(vecKeys);
probs.getElement(fvs_pos) = params.getScore(prodFV);
++fvs_pos;
}
}
}
int last;
::readObject(featFile, last);
if(last != -3) { cerr << "Error reading file." << endl; return 0; }

if(m_isLabeled) {
vector<unsigned int> nt_dim(4);
unsigned int nt_pos;
nt_fvs.setDemisionVal(nt_dim, length, m_vecTypes.size(), 2, 2);
nt_fvs.resize(nt_dim);
nt_probs.resize(nt_dim);
nt_fvs.setDemisionVal(nt_dim, 0, 0, 0, 0);
nt_fvs.getElement(nt_dim, nt_pos);
for(int w1 = 0; w1 < length; w1++) {
for(int t = 0; t < m_vecTypes.size(); t++) {
const string &type = m_vecTypes[t];
for(int ph = 0; ph < 2; ph++) {						
for(int ch = 0; ch < 2; ch++) {
FeatureVec &prodFV = nt_fvs.getElement(nt_pos);
vector<int> vecKeys;
::readObject(featFile, vecKeys);
//cerr << vecKeys.size() << " ";
prodFV.setKeys(vecKeys);
nt_probs.getElement(nt_pos) = params.getScore(prodFV);
++nt_pos;
}
}
}
}
::readObject(featFile, last);
if(last != -3) { cerr << "Error reading file." << endl; return 0; }
}

FeatureVec &fv = m_instance.fv;
vector<int> vecKeys;
::readObject(featFile, vecKeys);
fv.setKeys(vecKeys);

::readObject(featFile, last);
if(last != -4) { cerr << "Error reading file." << endl; return 0; }

readObject(featFile, m_instance);

::readObject(featFile, last);
if(last != -1) { cerr << "Error reading file." << endl; return 0; }

return &m_instance;
}
*/

/*
void DepPipe::writeObject(ofstream &outf, const DepInstance &instance) {
instance.writeObject(outf);
}

void DepPipe::readObject(ifstream &inf, DepInstance &instance) {
instance.readObject(inf);
}
*/

///////////////////////////////
// Features //
///////////////////////////////

//void addExtendedFeature(DepInstance pInstance, FeatureVec &fv) {}

