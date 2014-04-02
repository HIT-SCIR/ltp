#ifndef __LTP_POSTAGGER_PREPROCESS_H__
#define __LTP_POSTAGGER_PREPROCESS_H__

#include <iostream>
#include <vector>
#include "chartypes.hpp"

#if _WIN32
// disable auto-link feature in boost
#define BOOST_ALL_NO_LIB
#endif

#include "boost/regex.hpp"

namespace ltp {
namespace postagger {
namespace preprocess {

const int WORD_PUNC = strutils::chartypes::CHAR_PUNC;
const int WORD_ENG = strutils::chartypes::CHAR_PUNC+1;
const int WORD_DIGITS = strutils::chartypes::CHAR_PUNC+2;
const int WORD_OTHER = strutils::chartypes::CHAR_OTHER;

static boost::regex engpattern("(([A-Za-z]+)([\\-'\\.][A-Za-z]+)*)");
static boost::regex digitspattern("\\d+");

inline int wordtype(const std::string & word){
    if( boost::regex_match(word,engpattern) ){
        return WORD_ENG;
    }
    else if( boost::regex_match(word,digitspattern) ){
        return WORD_DIGITS;
    }
    return strutils::chartypes::chartype(word);
}

}     //  end for preprocess
}     //  end for namespace postagger
}     //  end for namespace ltp 

#endif  //  end for __LTP_POSTAGGER_PREPROCESS_H__
