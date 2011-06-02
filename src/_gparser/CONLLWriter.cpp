#include "CONLLWriter.h"

#include <sstream>
using namespace std;

CONLLWriter::CONLLWriter()
{
}

CONLLWriter::~CONLLWriter()
{
}

int CONLLWriter::write(const DepInstance *pInstance)
{
	if (!m_outf.is_open()) return -1;

	const vector<string> &forms = pInstance->forms;
	const vector<string> &lemmas = pInstance->lemmas;
	const vector<string> &cpostags = pInstance->cpostags;
	const vector<string> &postags = pInstance->postags;
	const vector< vector<string> > &feats = pInstance->feats;
	const vector<string> &deprels = pInstance->deprels;
	const vector<int> &heads = pInstance->heads;

	const vector<double> &k_probs = pInstance->k_probs;
	const vector< vector<int> > &k_heads = pInstance->k_heads;
	const vector< vector<string> > &k_deprels = pInstance->k_deprels;

	if (!k_probs.empty()) {
		m_outf << k_probs[0];
		int i = 1;
		for (; i < k_probs.size(); ++i) {
			m_outf << "\t" << k_probs[i];
		}
		m_outf << endl;
	}

	int i = 1;
	for (; i < forms.size(); ++i) {
		m_outf << i << "\t"
				<< forms[i] << "\t"
				<< lemmas[i] << "\t"
				<< cpostags[i] << "\t"
				<< postags[i] << "\t"
				<< "_\t"
				<< heads[i] << "\t"
				<< deprels[i] << "\t"
				<< "_\t_";
		if (!k_probs.empty()) {
			int k = 0;
			for (; k < k_heads.size() && k < k_deprels.size(); ++k) {
				int k_head_i = i < k_heads[k].size() ? k_heads[k][i] : -1;
				string k_deprel_i = i < k_deprels[k].size() ? k_deprels[k][i] : "ERR";
				m_outf << "\t" << k_head_i << "\t" << k_deprel_i;
			}
		}
		m_outf << endl;

	}
	m_outf << endl;
	return 0;
}

