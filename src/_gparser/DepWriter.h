#ifndef _DEP_WRITER_
#define _DEP_WRITER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "DepInstance.h"

class DepWriter
{
public:
	DepWriter();
	virtual ~DepWriter();
	int startWriting(const char *filename) {
		m_outf.open(filename);
		if (!m_outf) {
			cerr << "DepWriterr::startWriting() open file err: " << filename << endl;
			return -1;
		}
		return 0;
	}

	void finishWriting() {
		m_outf.close();
	}

	virtual int write(const DepInstance *pInstance) = 0;
protected:
	ofstream m_outf;
};

#endif

