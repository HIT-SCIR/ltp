#include "DepReader.h"

DepReader::DepReader(void)
{
}

DepReader::~DepReader(void)
{
	if (m_inf.is_open()) m_inf.close();
}

string DepReader::normalize(const string &str)
{
	return str;
}

