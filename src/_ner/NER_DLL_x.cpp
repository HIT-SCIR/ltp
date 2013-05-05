#include "NER_DLL.h"
#include <iostream>
#include <string>
#include <vector>
#include <string.h>
using namespace std;

static void split_bychar(const string& str, vector<string>& vec,
						 const char separator)
{
	//assert(vec.empty());
	vec.clear();
	string::size_type pos1 = 0, pos2 = 0;
	string word;
	while((pos2 = str.find_first_of(separator, pos1)) != string::npos)
	{
		word = str.substr(pos1, pos2-pos1);
		pos1 = pos2 + 1;
		if(!word.empty()) 
			vec.push_back(word);
	}
	word = str.substr(pos1);
	if(!word.empty())
		vec.push_back(word);
}

int NER(void* NETagger, const vector<string>& vecWord, const vector<string>& vecPOS, vector<string>& vecResult)
{
	string strin;
	int nChar = 0;
	int i = 0;
	for (; i<(int)vecWord.size(); ++i)
	{
		nChar += vecWord[i].length() + vecPOS[i].length();
		strin += vecWord[i] + "/" + vecPOS[i] + " ";
	}
	const int SZ = nChar + vecWord.size() * 32;
	char* presult = new char[SZ];
	memset(presult, 0, SZ);
/*
也/d 是/v 国内/nl SVM/ws 最好/d 的/u 学者/n 之/u 
一  /m 4/m 、/wp 数据/n 挖掘/v 中/nd 的/u 新/a 方法/n ：/wp 
*/

	NERtesting(NETagger, (char *)strin.c_str(), presult, 2); //进行NE识别
	string NEresult = presult;
	
	delete [] presult;
	
	//cout << NEresult << "||||";

/*
也/d#O 是/v#O 国内/nl#O SVM/ws#O 最好/d#O 
的/u#O 学者/n#O 之/u#O 一  /m 4/m 、/wp 数据/n 挖掘/v 中/nd 的/u 新/a 方法/n ：/wp /u 
一#O  /m 4/m 、/wp 数据/n 挖掘/v 中/nd 的/u 新/a 方法/n ：/wp /u 一 #O 
/m#B-Nm 4/m#E-Nm 、/wp#O 数据/n#O 挖掘/v#O 中/nd#O 的/u#O 新/a#O 方法/n#O ：/wp#O
*/
	vector<string> vecTmp;
	split_bychar(NEresult, vecTmp, ' ');
	if (vecTmp.size() != vecWord.size()) {
		cerr << strin << endl;
		cerr << NEresult << endl;
		cerr << "NE result word num != vecWord.size()" << endl;
		return -1;
	}

	for (i = 0; i < vecTmp.size(); ++i) {
		size_t pos = vecTmp[i].rfind('#');
		if (pos == string::npos) {
			vecResult.push_back("O");
		}
		else {
			vecResult.push_back( vecTmp[i].substr(pos+1, string::npos) );
		}
	}

	return 0;
}
