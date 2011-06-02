#include "DepWriter.h"

DepWriter::DepWriter()
{
}

DepWriter::~DepWriter()
{
	if (m_outf.is_open()) m_outf.close();
}
