#ifndef _ALPHABET_
#define _ALPHABET_
#pragma once

#include <map>
using namespace std;

#include "MyLib.h"

/*
	This class serializes feature from string to int.
*/

class Alphabet
{
public:
	Alphabet() {
		m_map.clear();
		m_numEntries = 0;
		m_isGrowthStopped = false;
	}
	~Alphabet(void) {}
	int lookupIndex(const string &str);
	int size() {
		return m_map.size();
	}
//	int add(const string &str);
/*	int add(const string &str, int idx) {
		m_map[str] = idx;
	}
*/	void allowGrowth() {
		m_isGrowthStopped = false;
	}
	void stopGrowth() {
		m_isGrowthStopped = true;
	}

	void show() {
		cerr << "total num: " << m_map.size() << endl;
		map<string, int, string_less>::const_iterator it = m_map.begin();
		while (it != m_map.end()) {
			cerr << "(" << it->first << " " << it->second << ")" << endl;
			++it;
		}
	}

	void getKeys(vector<string> &vecKeys) const;

	void readObject(ifstream &inf);

	void writeObject(ofstream &outf) const;

private:
	map<string, int, string_less> m_map;
	int m_numEntries;
	bool m_isGrowthStopped;
};

#endif

