/////////////////////////////////////////////////////////////////////////////////////
// File Name   : Dictionary.h
// Project Name: WTLAS
// Author      : Liqi Gao
// Environment : Linux/Windows
// Description :
// Time        : 2005.9
// History     :
// CopyRight   : Wintim (c) 2005-2007, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////
#ifndef __DICT_BASE_H__
#define __DICT_BASE_H__

#include "LASBase.h"

#include <map>
#include <ostream>
#include <string>

LAS_NS_BEG

// the base of dictionary
class DictBase
{
public:
#ifdef _WIN32
//	typedef __int64 value_type;
	typedef int value_type;
#else
//	typedef long long value_type;
	typedef int value_type;
#endif

	enum {
		default_freq = 100,
		not_found = -1
	};
	DictBase()	{};
	virtual ~DictBase() {};

	// get the total frequency of all words
	virtual value_type TotalFreq() const = 0;

	// tell if the dictionary contains the word
	virtual bool Has(const char *word) const = 0;
	virtual bool Has(const std::string &word) const = 0;

	virtual value_type WordFreq(const char *word) const = 0;

	// return the number of words
	virtual value_type WordCount() const = 0;

	// dump the whole dictionary into a stream
	virtual void Dump(std::ostream *out) = 0;

	// returns true if it is successfully loaded
	virtual bool IsLoaded() const
	{
		return bLoaded;
	}

protected:
	bool bLoaded;
	value_type  _default_freq;

};

LAS_NS_END

#endif
