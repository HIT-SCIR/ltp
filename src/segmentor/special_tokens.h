#ifndef __LTP_SPECIAL_TOKENS_H__
#define __LTP_SPECIAL_TOKENS_H__

#include <iostream>

namespace ltp{
namespace segmentor{
const static std::string special_tokens[] = {
"AT&T",
"c#",
"C#",
"c++",
"C++",
};

const static size_t special_tokens_size = sizeof(special_tokens) / sizeof(std::string);
}
}


#endif
