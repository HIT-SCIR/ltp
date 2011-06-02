#ifndef __SEGMENT_ENGINE_H__
#define __SEGMENT_ENGINE_H__

#include "LASBase.h"
#include "DictBase.h"
#include <vector>
#include <string>

LAS_NS_BEG

class SegEngine
{
protected:
	static const double inf;
	static const double epsilon;
	static const int 	MAX_WORD_LENGTH;

public:
	SegEngine();
	virtual ~SegEngine();

	virtual std::string ToString() = 0;

	virtual bool Segment(const char *pszText, DictBase *pDict, std::vector<std::string> &vecSegResult) = 0;
};

LAS_NS_END
#endif // __SEGMENT_ENGINE_H__
