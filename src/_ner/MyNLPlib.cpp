#include "MyNLPlib.h"
#include <math.h>

inline int getFileLength(FILE *fp)
{
	fseek(fp, 0, SEEK_END);     //find the end of the file
	int filelength = ftell(fp); 
	return filelength;
}


void encodeFile(string& infilename, string& outfilename)
{
	FILE *fpin = fopen(infilename.c_str(), "rb");
	FILE *fpout = fopen(outfilename.c_str(), "wb");

	if (fpin == NULL)
	{
		cerr << "Can not open the infile!" << endl;
		exit(-1);
	}

	if (fpout == NULL)
	{
		cerr << "Can not open the outfile!" << endl;
		exit(-1);
	}
    
	fseek(fpin, 0, SEEK_END);
   	int fileLength = ftell(fpin);     //get the current position of a file pointer
	char* inbuf = new char[fileLength];
	fseek(fpin, 0, SEEK_SET);         //find the begin of the file
	fread(inbuf, fileLength, 1, fpin);
	fclose(fpin);

	for (int i=0; i<fileLength; i++)  //Add password
	{
		inbuf[i] ^= 0xAB;
	}
	
	fwrite(inbuf, fileLength, 1, fpout);
	fclose(fpout);
    delete [] inbuf;

///////////////////        说明        /////////////////////////////////
////下面的内容可以不要，其目的是为了验证加密是否正确，即检查是否可以还原
////////////////////////////////////////////////////////////////////////

//	FILE *fp;                   
//	fp = fopen("temp.txt", "w");
//	if (!fp)
//	{
//		cerr << "Can not open the file!" << endl;
//		exit(-1);
//	}
//	for (i=0; i<fileLength; i++)
//	{
//		inbuf[i] ^= 0xAB;
//	}
//
//	fwrite(inbuf, fileLength, 1, fp);
//	delete [] inbuf;
//	fclose(fp);
}


///////////////////////////////////////////////////////////////
//	Funcname : splitSentence
//	Caller   : 
//	Function : split Sentence by char ' ' 注意：结果中词性不含/
//	Author   : 
//	Time     : 2005-7-16
//	RetValue : void
//	Param    : const string& strSen,
//				 vector<string>& Con,
//				 char SplitChar
///////////////////////////////////////////////////////////////

void splitSenByWord(const string& strSen, vector< pair<string, string> >& Con, const char SplitChar)
{
	string::size_type pos1 = 0, pos2 = 0;
	string::size_type pos3 = 0;
	string strTemp = strSen;
	pair<string, string> paSen;

	if (strSen[strSen.size()-1] != ' ')
	{
		strTemp = strSen + " ";
	}

	while((pos1=strTemp.find_first_of(SplitChar, pos1)) != string::npos)
	{
		pos3 = strTemp.rfind("/", pos1);
		paSen.first = strTemp.substr(pos2, pos3-pos2);
		paSen.second = strTemp.substr(pos3+1, pos1-pos3-1);  //结果中不含有 /
		Con.push_back(paSen);
		pos2 = ++pos1;
	}
}


///////////////////////////////////////////////////////////////
//	Funcname : getNEtypeindex
//	Caller   : 
//	Function : get NEtype index
//	Author   : 
//	Time     : 2005-7-16
//	RetValue : int
//	Param    : string& NEtype
///////////////////////////////////////////////////////////////

inline int getNEtypeindex(string& NEtype)
{
	if (NEtype.size() == 2)
	{
		switch(NEtype[1])
			{
			case 'h': return 0; //nh
			case 's': return 1; //ns
			case 'i': return 2; //ni
			case 'z': return 3; //nz
			case 't': return 4; //nt
			case 'r': return 5; //nr
			case 'u': return 6; //nu
			default: return -1;
			}	
	}
	else 
	{
		return -1;
	}
}


///////////////////////////////////////////////////////////////
//	Funcname : splitSenByNE
//	Caller   : 
//	Function : 
//	Author   : 
//	Time     : 2005-7-16
//	RetValue : void
//	Param    : const string& strSen,
//				 vector< pair<string,
//				 string> >& vecOut
///////////////////////////////////////////////////////////////

void splitSenByNE(const string& strSen, vector< pair<string, string> >& vecOut)
{
	string::size_type pos1 = 0;
	string::size_type pos2 = 0;
	string::size_type pos3 = 0;
	pair<string, string> paSen;

	string strTemp = strSen;
	string strPOS;

	if (strSen[strSen.size()-1] != ' ')
	{
		strTemp = strTemp + " ";
	}

	while ((pos3=strTemp.find_first_of(" ", pos1)) != string::npos)
	{
		if (strTemp[pos1] != '[')
		{
			pos2 = strTemp.rfind("/", pos3);
			paSen.first = strTemp.substr(pos1, pos2-pos1);
            strPOS = strTemp.substr(pos2, pos3-pos2);
			paSen.second = strPOS.substr(0, strPOS.find("]"));
			//paSen.second = strTemp.substr(pos2, pos3-pos2);
			pos1 = pos3 + 1;
			//pos3 = pos1;
			vecOut.push_back(paSen);
		}
		else
		{
			pos2 = strTemp.find("]", pos1);
			if (pos2 != string::npos)
			{
				pos3 = strTemp.find(" ", pos2);
				if (pos3 != string::npos)
				{
				    if (strTemp.substr(pos2+1, pos3-pos2-1).size() != 2)
					{
						pos1 = pos1 + 1;
						continue;
					}
					else
					{
						paSen.first = strTemp.substr(pos1, pos3-pos1);
						paSen.second = "";
						pos1 = pos3 + 1;
					}					
					//pos3 = pos1;
				}
				else
				{
					if (strTemp.substr(pos2+1).size() != 2)
					{
						pos1 = pos1 + 1;
						continue;
					}
					else
					{
						paSen.first = strTemp.substr(pos1);
						paSen.second = "";
						pos1 = strTemp.size();
					}
				}
				vecOut.push_back(paSen);
			}
			else
			{
				pos1 = pos1 + 1;
			}
		}
	}	
}

void splitSenByChar(const string& strSen, const char spliter, vector<string>& vecOut)
{
	string::size_type pos1 = 0;
	string::size_type pos2 = 0;
	string strTemp;

	while ((pos2=strSen.find_first_of(spliter, pos1)) != string::npos)
	{
		strTemp = strSen.substr(pos1, pos2-pos1);
		vecOut.push_back(strTemp);
		pos1 = pos2 + 1;
	}
	if (pos1 < strSen.size())
	{
		strTemp = strSen.substr(pos1);
		vecOut.push_back(strTemp);
	}
}

///////////////////////////////////////////////////////////////
//	Funcname : splitSenByNETag
//	Caller   : 
//	Function : 
//	Author   : 
//	Time     : 2005-7-21
//	RetValue : void
//	Param    : const string& strSen,
//			   vector<NENODE>& vecOut
///////////////////////////////////////////////////////////////

void splitSenByNETag(const string& strSen, vector<NENODE>& vecOut)
{
	NENODE Ner;
	vector< pair<string, string> > vec2paSenNE;
	vector< pair<string, string> > vec2paSenWord;
	splitSenByNE(strSen, vec2paSenNE);
	//showvec2pairContent(vec2paSenNE);

	int size = 0; 
	int NElength = 0;
	string NEtype;
	
	for (int i=0; i<(int)vec2paSenNE.size(); i++)
	{
		if (vec2paSenNE[i].second != "")
		{
			Ner.Word = vec2paSenNE[i].first;
			Ner.Pos = vec2paSenNE[i].second.substr(1);
			Ner.NEtag = "O";
			vecOut.push_back(Ner);
		}
		else
		{
			size = (int)vec2paSenNE[i].first.size();
			string temp = vec2paSenNE[i].first.substr(1, size-4);
			splitSenByWord(vec2paSenNE[i].first.substr(1, size-4), vec2paSenWord, ' ');  //take note
			
			//showvec2pairContent(vec2paSenWord); //for debug

			NEtype = vec2paSenNE[i].first.substr(size-2);

			NEtagBIO(vec2paSenWord, NEtype, vecOut);
			//NEtagBIESO(vec2paSenWord, NEtype, vecOut);
			vec2paSenWord.clear();
		}
	}
	vec2paSenNE.clear();
}


///////////////////////////////////////////////////////////////
//	Funcname : NEtagBIO
//	Caller   : 
//	Function : 
//	Author   : 
//	Time     : 2005-7-21
//	RetValue : void
//	Param    : const vector< pair<string,
//				 string> >& vecIn,
//				        const string& NEtype,
//				 vector<NENODE>& vecOut
///////////////////////////////////////////////////////////////

void NEtagBIO(const vector< pair<string, string> >& vecIn, 
			  const string& NEtype, vector<NENODE>& vecOut)
{
	NENODE Ner;
	int NElength = (int)vecIn.size();

	Ner.Word = vecIn[0].first;
	Ner.Pos = vecIn[0].second;
	Ner.NEtag = "B-" + NEtype;
	vecOut.push_back(Ner);

	for (int i=1; i<NElength; i++)
	{
		Ner.Word = vecIn[i].first;
		Ner.Pos = vecIn[i].second;
		Ner.NEtag = "I-" + NEtype;
		vecOut.push_back(Ner);
	}	
}


///////////////////////////////////////////////////////////////
//	Funcname : NEtagBIESO
//	Caller   : 
//	Function : 
//	Author   : 
//	Time     : 2005-7-21
//	RetValue : void
//	Param    : const vector< pair<string,
//				 string> >& vecIn,
//				       const string& NEtype,
//				 vector<NENODE>& vecOut
///////////////////////////////////////////////////////////////

void NEtagBIESO(const vector< pair<string, string> >& vecIn,
			  const string& NEtype, vector<NENODE>& vecOut)
{
	NENODE Ner;
	int NElength = (int)vecIn.size();
	
	if (NElength == 1)  //NE-single
	{
		Ner.Word = vecIn[0].first;
    	Ner.Pos = vecIn[0].second;
		Ner.NEtag = "S-" + NEtype;
		vecOut.push_back(Ner);
	}
	else
	{
		Ner.Word = vecIn[0].first;
		Ner.Pos = vecIn[0].second;
		Ner.NEtag = "B-" + NEtype;
		vecOut.push_back(Ner);   //NE-begin

		int i;

		for (i=1; i<NElength-1; i++)  //NE-inside
		{
			Ner.Word = vecIn[i].first;
			Ner.Pos = vecIn[i].second;
			Ner.NEtag = "I-" + NEtype;
			vecOut.push_back(Ner);
		}
		
		Ner.Word = vecIn[i].first;  //NE-end
		Ner.Pos = vecIn[i].second;
		Ner.NEtag = "E-" + NEtype;
		vecOut.push_back(Ner);
	}	
}
///////////////////////////////////////////////////////////////
//	Funcname : makeaMapForRule
//	Caller   : 
//	Function : 
//	Author   : 
//	Time     : 2005-7-17
//	RetValue : void
//	Param    : const string& filename,
//				 map<string,
//				 int>& mapName,
//				 int mapValue
///////////////////////////////////////////////////////////////

void makeaMapForRule(const string& filename, map<string, int>& mapName, int mapValue)
{
	ifstream inmap(filename.c_str());

	string strIn;

	while (getline(inmap, strIn))
	{
		if (!strIn.empty())
		{
			mapName[strIn] = mapValue;
		}		
	}

	inmap.close();
}

inline void showvectorContent(vector<string>& vecTemp)
{
	//ofstream out1("vecString.txt");
	vector<string>::iterator iter;
	for (iter=vecTemp.begin(); iter!=vecTemp.end(); ++iter)
	{
		cout << *iter << endl;
		//out1 << *iter << " ";
	}
	//out1.close();
}

inline void showvec2pairContent(vector< pair<string, string> >& vec2paTemp)
{
	//ofstream out2("vec2pa.txt");
	vector< pair<string, string> >::iterator iter;
	for(iter=vec2paTemp.begin(); iter!=vec2paTemp.end(); ++iter)
	{
		cout << iter->first << " " << iter->second << endl; 
		//out2 << iter->first << " " << iter->second << endl;
	}

	//out2.close();
}

inline void showvec2paContent2(vector< pair<string, double> >& vec2paTemp, ofstream& outfile)
{
	vector< pair<string, double> >::iterator iter;
	for(iter=vec2paTemp.begin(); iter!=vec2paTemp.end(); ++iter)
		{
			outfile << iter->first << " " << -log(iter->second) << endl; 
			cout << iter->first << " " << -log(iter->second) << endl; 
		}
}


///////////////////////////////////////////////////////////////
//	Funcname : sortWithUniqueElement
//	Caller   : 
//	Function : 
//	Author   : Tina Liao
//	Time     : 2005-8-3
//	RetValue : void
//	Param    : ifstream& infile,
//				 ofstream& outfile
///////////////////////////////////////////////////////////////

void sortWithUniqueElement(ifstream& infile, ofstream& outfile)
{
	if (!infile)
	{
		cerr << "Can not open the infile!" << endl;
		exit(-1);
	}

	vector<string> coll;
	copy(istream_iterator<string>(infile), istream_iterator<string>(), back_inserter(coll));
	sort(coll.begin(), coll.end());
	unique_copy(coll.begin(), coll.end(), ostream_iterator<string>(outfile, "\n"));
	coll.clear();
}

///////////////////////////////////////////////////////////////
//	Funcname : extractSen
//	Caller   : 
//	Function : 用于根据具体的位置信息提取子串
//	Author   : 
//	Time     : 2005-9-16
//	RetValue : string
//	Param    : string& Src,
//				 POSITION begin,
//				 POSITION end
///////////////////////////////////////////////////////////////

inline string extractSen(string& Src, POSITION begin, POSITION end)
{
	string Result;
	if (end != string::npos)
	{
		Result = Src.substr(begin, end-begin);
	}
	else
	{
		Result = Src.substr(begin);
	}
	return Result;
}


string ArrNEtype1[9] = {"B-Nh", "B-Ni", "B-Ns", "B-Nz",
                        "I-Nh", "I-Ni", "I-Ns", "I-Nz",
                        "O"};
string ArrNEtype2[5] = {"B-Nt", "B-Nr",
						"I-Nt", "I-Nr",
						"O"};
string ArrNEtype3[3] = {"B-Nm",
						"I-Nm",
						"O"};

bool isNEtype(string& strIn, bool* bisNEtypeflag)
{
	if (bisNEtypeflag[2])
	{
		if (find(&ArrNEtype3[0], &ArrNEtype3[3], strIn) != &ArrNEtype3[3])
		{
			return true;
		}
	}
	if (bisNEtypeflag[1])
	{
		if (find(&ArrNEtype2[0], &ArrNEtype2[5], strIn) != &ArrNEtype2[5])
		{
			return true;
		}
	}
	if (bisNEtypeflag[0])
	{
		if (find(&ArrNEtype1[0], &ArrNEtype1[9], strIn) != &ArrNEtype1[9])
		{
			return true;
		}
	}
	return false;
}