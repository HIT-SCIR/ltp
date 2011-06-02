#include "DepInstance.h"
/*
void DepInstance::writeObject(ofstream &outf) const 
{
	outf << "[inst]" << endl;
	copy(forms.begin(), forms.end(), ostream_iterator<string>(outf, "\t"));
	outf << endl;
	copy(lemmas.begin(), lemmas.end(), ostream_iterator<string>(outf, "\t"));
	outf << endl;
	copy(cpostags.begin(), cpostags.end(), ostream_iterator<string>(outf, "\t"));
	outf << endl;
	copy(postags.begin(), postags.end(), ostream_iterator<string>(outf, "\t"));
	outf << endl;
	int i = 0;
	for (; i < feats.size(); ++i) {
		copy(feats[i].begin(), feats[i].end(), ostream_iterator<string>(outf, "|"));
		outf << "\t";
	}
	outf << endl;
	copy(heads.begin(), heads.end(), ostream_iterator<int>(outf, "\t"));
	outf << endl;
	copy(deprels.begin(), deprels.end(), ostream_iterator<string>(outf, "\t"));
	outf << endl;
	outf << actParseTree << endl;
	outf << endl;
}

void DepInstance::readObject(ifstream &inf) 
{
	string strLine;
	my_getline(inf, strLine);
	if (strLine != "[inst]") {
		cerr << "DepInstance::readObject() err: " << strLine << endl;
		return;
	}
	my_getline(inf, strLine);
	split_bychar(strLine, forms, '\t');
	my_getline(inf, strLine);
	split_bychar(strLine, lemmas, '\t');
	my_getline(inf, strLine);
	split_bychar(strLine, cpostags, '\t');
	my_getline(inf, strLine);
	split_bychar(strLine, postags, '\t');

	vector<string> vec;
	my_getline(inf, strLine);
	split_bychar(strLine, vec, '\t');
	feats.resize(vec.size());
	int i = 0;
	for (; i < vec.size(); ++i) {
		split_bychar(vec[i], feats[i], '|');
	}

	my_getline(inf, strLine);
	split_bychar(strLine, vec, '\t');
	str2int_vec(vec, heads);

	my_getline(inf, strLine);
	split_bychar(strLine, deprels, '\t');

	my_getline(inf, actParseTree);
	my_getline(inf, strLine);
}
*/


