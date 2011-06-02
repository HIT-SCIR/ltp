#include "SplitSentence.h"
#include "SentenceIterator.h"
#include <string.h>


using namespace util;
using namespace Chinese;

int SplitSentence( const string& strPara, vector<string>& vecSentence )
{
	StringReader sr( strPara.c_str() );
	Separator sep;
	sentence_iterator si(&sr, sep), send;
	vecSentence.clear();
	while( si != send )
	{
		if(strlen(*si) < 400){
			vecSentence.push_back( *si );
		}
		si++;
	}
	return 1;
}
