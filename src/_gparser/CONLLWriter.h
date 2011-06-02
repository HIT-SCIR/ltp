#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#pragma once
#include "DepWriter.h"

/*
	this class writes conll-format result (no srl-info).
*/
class CONLLWriter : public DepWriter
{
public:
	CONLLWriter();
	~CONLLWriter();
	int write(const DepInstance *pInstance);
};

#endif

