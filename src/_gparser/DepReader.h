#ifndef _DEP_READER_
#define _DEP_READER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "DepInstance.h"

class DepReader
{
public:
	DepReader();
	virtual ~DepReader();
	int startReading(const char *filename) {
		if (m_inf.is_open()) {
/*			cerr << endl;
			cerr << ( m_inf.rdstate( ) & ios::badbit ) << endl;
			cerr << ( m_inf.rdstate( ) & ios::failbit ) << endl;
			cerr << ( m_inf.rdstate( ) & ios::eofbit ) << endl;
			cerr << m_inf.good() << endl;
			cerr << m_inf.bad() << endl;
			cerr << m_inf.fail() << endl;
			cerr << m_inf.eof() << endl;
			cerr << endl;
*/			m_inf.close();
			m_inf.clear();
		}
		m_inf.open(filename);
/*		cerr << endl;
		cerr << ( m_inf.rdstate( ) & ios::badbit ) << endl;
		cerr << ( m_inf.rdstate( ) & ios::failbit ) << endl;
		cerr << ( m_inf.rdstate( ) & ios::eofbit ) << endl;
		cerr << m_inf.good() << endl;
		cerr << m_inf.bad() << endl;
		cerr << m_inf.fail() << endl;
		cerr << m_inf.eof() << endl;
		cerr << endl;
*/		if (!m_inf.is_open()) {
			cerr << "DepReader::startReading() open file err: " << filename << endl;
			return -1;
		} 
//		m_inf.seekg(0, ios_base::beg);
		return 0;
	}

	void finishReading() {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
	}

	string normalize(const string &str);
	virtual DepInstance *getNext() = 0;
protected:
	ifstream m_inf;
//	bool m_isLabeled;
	int m_numInstance;

	DepInstance m_instance;
};

#endif

