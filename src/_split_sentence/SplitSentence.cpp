#include "SplitSentence.h"
//#include "SentenceIterator.h"
#include <string.h>
#include "sentsplit.hpp"

//using namespace util;

int SplitSentence( const std::string& strPara, std::vector<std::string>& vecSentence ) {
    ltp::Chinese::split_sentence(strPara, vecSentence);
    /*StringReader sr( strPara.c_str() );
    Separator sep;
    sentence_iterator si(&sr, sep), send;
    vecSentence.clear();
    while( si != send ) {
        if(strlen(*si) < 400){
            vecSentence.push_back( *si );
        }
        si++;
    }*/
    return 1;
}
