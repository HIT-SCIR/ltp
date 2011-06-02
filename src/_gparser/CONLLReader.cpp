#include "CONLLReader.h"
#include "MyLib.h"

#include <sstream>
using namespace std;

CONLLReader::CONLLReader(void)
{
}

CONLLReader::~CONLLReader(void)
{
}

DepInstance *CONLLReader::getNext()
{
	m_instance.fv.clear();
	m_instance.forms.clear();

	vector<string> vecLine;
	while (1) {
		string strLine;
		if (!my_getline(m_inf, strLine)) {
/*			cerr << endl;
			cerr << ( m_inf.rdstate( ) & ios::badbit ) << endl;
			cerr << ( m_inf.rdstate( ) & ios::failbit ) << endl;
			cerr << ( m_inf.rdstate( ) & ios::eofbit ) << endl;
			cerr << m_inf.good() << endl;
			cerr << m_inf.bad() << endl;
			cerr << m_inf.fail() << endl;
			cerr << m_inf.eof() << endl;
			cerr << endl;
*/			break;
		}
		if (strLine.empty() || strLine[0] == '*') break;
		vecLine.push_back(strLine);	
	}
	if (vecLine.empty()) {
		//m_inf.close();
//		cerr << "CONLLReader::getNext() : vecLine is empty" << endl;
		return 0;
	}

    int length = vecLine.size();
	vector<string> &forms = m_instance.forms;
	vector<string> &lemmas = m_instance.lemmas;
	vector<string> &cpos = m_instance.cpostags;
	vector<string> &pos = m_instance.postags;
	vector< vector<string> > &feats = m_instance.feats;
	vector<string> &deprels = m_instance.deprels;
	vector<int> &heads = m_instance.heads;
	forms.resize(length+1);
	lemmas.resize(length+1);
	cpos.resize(length+1);
	pos.resize(length+1);
	feats.resize(length+1);
	deprels.resize(length+1);
	heads.resize(length+1);

	forms[0] = "<root>";
	lemmas[0] = "<root-LEMMA>";
	cpos[0] = "<root-CPOS>";
	pos[0] = "<root-POS>";
	deprels[0] = "<no-type>";
	heads[0] = -1;

	int i = 0;
	for (; i < length; ++i) {
		const string &strLine = vecLine[i];
		vector<string> vecInfo;
		split_bychar(strLine, vecInfo, '\t');
		forms[i+1] = normalize(vecInfo[1]);
		lemmas[i+1] = normalize(vecInfo[2]);
		cpos[i+1] = vecInfo[3];
		pos[i+1] = vecInfo[4];
		split_bychar(vecInfo[5], feats[i+1], '|');
		deprels[i+1] = vecInfo[7];
		heads[i+1] = atoi(vecInfo[6].c_str());
	}

	feats[0].resize( feats[1].size() );
	for (i = 0; i < feats[1].size(); ++i) {
		ostringstream out;
		out << "<root-feat>" << i;
		feats[0][i] = out.str();
	}
	return &m_instance;
//	m_instance.setInstance(forms, lemmas, cpos, pos, feats, deprels, heads);
}

