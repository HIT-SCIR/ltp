#ifndef __SPLIT_SENTENCE_H__
#define __SPLIT_SENTENCE_H__

#pragma warning(disable: 4786)

#include <string>
#include <vector>

// return (int)vecSentence.size();
int SplitSentence( const std::string& strPara, std::vector<std::string>& vecSentence );

#endif //__SPLIT_SENTENCE_H__