#include "splitsnt/SplitSentence.h"
#include "utils/strutils.hpp"
#include "utils/sentsplit.hpp"

//using namespace util;

int SplitSentence( const std::string& strPara, std::vector<std::string>& vecSentence ) {
    ltp::Chinese::split_sentence(strPara, vecSentence);

    for (int i = 0; i < vecSentence.size(); ++ i) {
        vecSentence[i] = ltp::strutils::chomp(vecSentence[i]);
    }

    return 1;
}
