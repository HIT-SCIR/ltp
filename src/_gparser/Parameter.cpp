#include "Parameter.h"
#include "MyLib.h"
#include "MultiArray.h"

const double EPS = 0.00000001;
const double ZERO = 0.000000000001;
const double DOUBLE_NEGATIVE_INFINITY = - 10e20;

void Parameter::updateParamsMIRA(DepInstance *pInstance, vector<FeatureVec> &d0, vector<string> &d1, double upd)
{
	const string &actParseTree = pInstance->actParseTree;
	FeatureVec &actFV = pInstance->fv;

	int K = 0;
	int i = 0;
	for(; i < d0.size() && i < d1.size() && !d1[i].empty(); i++) {
		K = i+1;
	}

	vector<double> b(K, 0.0);
	vector<double> lam_dist(K, 0.0);
	vector<FeatureVec> dist(K);
	int k = 0;
	int _K = K;
	for(; k < K; k++) {
		if (d1[k].empty()) {
			_K = k;
			break;
		}
		lam_dist[k] = getScore(actFV) - getScore(d0[k]);
		b[k] = (double)numErrors(pInstance, d1[k], actParseTree);

/*		cerr << "score dist: " << lam_dist[k] << endl;
		cerr << "err num: " << b[k] << endl;
		cerr << "sys parse: " << d1[k] << endl;
		cerr << "gold parse: " << actParseTree << endl;
		vector<int> vecKeys;
		actFV.getKeys(vecKeys);
		cerr << "gold parse feat num: " << vecKeys.size() << endl;
		d0[k].getKeys(vecKeys);
		cerr << "sys parse feat num: " << vecKeys.size() << endl;
*/		
		b[k] -= lam_dist[k];
		dist[k].add(&actFV);
		dist[k].remove(&d0[k]);
//		dist[k].collectFeatures();

/*		vector<int> vecKeys;
		dist[k].getKeys(vecKeys);
		cerr << "dist vec key num: " << vecKeys.size() << endl;
		d0[k].getKeys(vecKeys);
		cerr << "d0[k] vec key num: " << vecKeys.size() << endl;
		actFV.getKeys(vecKeys);
		cerr << "gold tree vec key num: " << vecKeys.size() << endl;
*/	}

	vector<double> alpha; 
//	cerr << "(";
	hildreth(dist, b, alpha, _K);
//	cerr << ")";

	int res = 0;
	for(k = 0; k < _K; k++) {
		//		for(k = 0; k < K; k++) {
//		cerr << "[" << alpha[k] << "]" << endl;
		dist[k].update(m_parameters, m_total, alpha[k], upd);
//		cerr << "\n-----\n" << endl;
	}
}

void Parameter::hildreth(const vector<FeatureVec> &a, const vector<double> &b, vector<double> &alpha, int K) 
{
	int max_iter = 10000;
	alpha.resize(K);
	vector<double> F(K, 0.0);
	vector<double> kkt(K, 0.0);
	double max_kkt = DOUBLE_NEGATIVE_INFINITY;

	static vector<unsigned int> A_dim;
	unsigned int A_pos;
	MultiArray<double> A;
	A.setDemisionVal(A_dim, K, K);
	A.resize(A_dim, 0.0);

	vector<bool> is_computed(K, false);

	int i;
	for(i = 0; i < K; i++) {
		A.setDemisionVal(A_dim, i, i);
		A.getElement(A_dim, A_pos) = FeatureVec::dotProduct(a[i], a[i]);
	}

	int max_kkt_i = -1;

	for(i = 0; i < F.size(); i++) {
		F[i] = b[i];
		kkt[i] = b[i];
		if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
	}

	int iter = 0;
	double diff_alpha;
	double try_alpha;
	double add_alpha;

	while(max_kkt >= EPS && iter < max_iter) {
//		cerr << ".";

		A.setDemisionVal(A_dim, max_kkt_i, max_kkt_i);
		A.getElement(A_dim, A_pos);

		diff_alpha = A.getElement(A_pos) <= ZERO ? 0.0 : F[max_kkt_i]/A.getElement(A_pos);
		try_alpha = alpha[max_kkt_i] + diff_alpha;
		add_alpha = 0.0;

		if(try_alpha < 0.0)
			add_alpha = -1.0 * alpha[max_kkt_i];
		else
			add_alpha = diff_alpha;

		alpha[max_kkt_i] = alpha[max_kkt_i] + add_alpha;

		if (!is_computed[max_kkt_i]) {
			for(i = 0; i < K; i++) {
				A.setDemisionVal(A_dim, i, max_kkt_i);
				A.getElement(A_dim, A_pos) = FeatureVec::dotProduct(a[max_kkt_i], a[i]); // for version 1
				is_computed[max_kkt_i] = true;
			}
		}

		for(i = 0; i < F.size(); i++) {
			A.setDemisionVal(A_dim, i, max_kkt_i);			
			F[i] -= add_alpha * A.getElement(A_dim, A_pos);
			kkt[i] = F[i];
			if(alpha[i] > ZERO)
				kkt[i] = abs(F[i]);
		}

		max_kkt = DOUBLE_NEGATIVE_INFINITY;
		max_kkt_i = -1;
		for(i = 0; i < F.size(); i++) {
			if(kkt[i] > max_kkt) { max_kkt = kkt[i]; max_kkt_i = i; }
		}
		iter++;
	}
}

double Parameter::numErrorsArc(DepInstance *pInstance, const string &pred, const string &act) 
{
	vector<string> act_spans;
	split_bychar(act, act_spans, ' ');
	vector<string> pred_spans;
	split_bychar(pred, pred_spans, ' ');
	int correct = 0;
	int i = 0;
	for(; i < pred_spans.size() && i < act_spans.size(); i++) {
		vector<string> vec;
		vector<string> vec2;
		split_bychar(pred_spans[i], vec, ':');
		split_bychar(act_spans[i], vec2, ':');

		if (vec.empty() || vec2.empty()) {
			cerr << "span format err: " << pred_spans[i] << " : " << act_spans[i] << endl;
			continue;
		}
		if(vec[0] == vec2[0]) {
			correct++;
		}
	}

	return ((double)act_spans.size() - correct);
}

double Parameter::numErrorsLabel(DepInstance *pInstance, const string &pred, const string &act) 
{
	vector<string> act_spans;
	split_bychar(act, act_spans, ' ');
	vector<string> pred_spans;
	split_bychar(pred, pred_spans, ' ');
/*	cerr << endl;
	copy(pred_spans.begin(), pred_spans.end(), ostream_iterator<string>(cerr, "#"));
	cerr << endl;
	copy(act_spans.begin(), act_spans.end(), ostream_iterator<string>(cerr, "#"));
	cerr << endl;
*/
	int correct = 0;
	int i = 0;
	for(; i < pred_spans.size() && i < act_spans.size(); i++) {
		vector<string> vec;
		vector<string> vec2;
		split_bychar(pred_spans[i], vec, ':');
		split_bychar(act_spans[i], vec2, ':');

		if (vec.size() < 2 || vec2.size() < 2) {
			cerr << "span format err: " << pred_spans[i] << " : " << act_spans[i] << endl;
			continue;
		}
		if (vec[0] == vec2[0]) {
			++correct;
		}
		if(vec[1] == vec2[1]) {
			++correct;
		}
	}
	return ((double)act_spans.size()*2 - correct);
}

double Parameter::numErrorsArcNoPunc(DepInstance *pInstance, const string &pred, const string &act) 
{
	vector<string> act_spans;
	split_bychar(act, act_spans, ' ');
	vector<string> pred_spans;
	split_bychar(pred, pred_spans, ' ');

	const vector<string> &forms = pInstance->forms;

	int correct = 0;
	int numPunc = 0;
	int i = 0;
	for(int i = 0; i < pred_spans.size(); i++) {
		vector<string> vec;
		vector<string> vec2;
		split_bychar(pred_spans[i], vec, ':');
		split_bychar(act_spans[i], vec2, ':');

		if (vec.empty() || vec2.empty()) {
			cerr << "span format err: " << pred_spans[i] << " : " << act_spans[i] << endl;
			continue;
		}

		if( isPunctuation(forms[i+1])) {
			numPunc++;
			continue;
		}

		if(vec[0] == vec2[0]) {
			correct++;
		}
	}

	return ((double)pred_spans.size() - numPunc - correct);
}

double Parameter::numErrorsLabelNoPunc(DepInstance *pInstance, const string &pred, const string &act) 
{
	vector<string> act_spans;
	split_bychar(act, act_spans, ' ');
	vector<string> pred_spans;
	split_bychar(pred, pred_spans, ' ');

	const vector<string> &forms = pInstance->forms;

	int correct = 0;
	int numPunc = 0;
	int i = 0;
	for(int i = 0; i < pred_spans.size(); i++) {
		vector<string> vec;
		vector<string> vec2;
		split_bychar(pred_spans[i], vec, ':');
		split_bychar(act_spans[i], vec2, ':');

		if (vec.size() < 2 || vec2.size() < 2) {
			cerr << "span format err: " << pred_spans[i] << " : " << act_spans[i] << endl;
			continue;
		}

		if( isPunctuation(forms[i+1])) {
			numPunc++;
			continue;
		}

		if (vec[0] == vec2[0]) {
			++correct;
			if(vec[1] == vec2[1]) {
				++correct;
			}
		}
	}

	return ((double)pred_spans.size()*2 - numPunc*2 - correct);
}

