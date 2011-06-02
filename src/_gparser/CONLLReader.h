#ifndef _CONLL_READER_
#define _CONLL_READER_

#pragma once
#include "DepReader.h"

/*
	this class reads conll-format data (10 columns, no srl-info)
*/
class CONLLReader : public DepReader
{
public:
	CONLLReader();
	~CONLLReader();

	DepInstance *getNext();
};

#endif

