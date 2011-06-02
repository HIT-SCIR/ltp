#include "Alphabet.h"

void Alphabet::getKeys(vector<string> &vecKeys) const 
{
	vecKeys.clear();
	map<string, int, string_less>::const_iterator it = m_map.begin();
	while (it != m_map.end()) {
		vecKeys.push_back(it->first);
		++it;
	}
}

void Alphabet::readObject(ifstream &inf) 
{
	int tmp;
	::readObject(inf, tmp);
	m_isGrowthStopped = tmp == 0 ? false : true;
	::readObject(inf, m_numEntries);
	int i = 0;
	for (; i < m_numEntries; ++i) {
		string strFeat;
		my_getline(inf, strFeat);
		string strIdx;
		my_getline(inf, strIdx);
		m_map[strFeat] = atoi(strIdx.c_str());
	}
}

void Alphabet::writeObject(ofstream &outf) const
{
	::writeObject(outf, (m_isGrowthStopped ? int(1) : int(0)));
	::writeObject(outf, m_numEntries);
	map<string, int, string_less>::const_iterator it = m_map.begin();
	while (it != m_map.end()) {
		outf << it->first << endl
			<< it->second << endl;
		++it;
	}
}

int Alphabet::lookupIndex(const string &str)
{
	map<string, int, string_less>::const_iterator it = m_map.find(str);
	if (it != m_map.end()) return it->second;
	if (m_isGrowthStopped) {
		return -1;
	} else {
		m_map[str] = m_numEntries;
//		cerr << "(" << str << " " << m_numEntries << ")" << endl;
		++m_numEntries;
		return (m_numEntries-1);
	}
}

/*
int Alphabet::add(const string &str)
{
	int idx = lookupIndex(str);
	if (idx == -1) {
		++m_numEntries;
		string str2 = str;
		m_map[str] = m_numEntries;
		return m_numEntries;
	} else {
		return idx;
	}
}
*/


