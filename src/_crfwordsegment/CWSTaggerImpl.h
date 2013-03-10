#ifndef __CWS_TAGGER_IMPL_H__
#define __CWS_TAGGER_IMPL_H__

#include "tagger.h"
//#include "../__crf++/tagger.h"
//#include <param.h>
#include <iostream>
#include <string>
#include <vector>

namespace CRFPP {
class CWSTaggerImpl : public TaggerImpl
{
protected:
	bool add(const std::string &line);
	const char* toString();
	void toString(std::vector<std::string> &vec);

public:
	bool read(std::istream *is);
	bool parse_stream(std::istream *is, std::ostream *os);
	bool parse_stream(const std::string &input, std::vector<std::string>& vecWords);

	CWSTaggerImpl();
	~CWSTaggerImpl();
};

}; // namespace CRFPP

#endif // __CWS_TAGGER_IMPL_H__
