#include "../__ltp_dll/__ltp_dll.h"
#pragma comment(lib, "__ltp_dll.lib")
#include "../__util/MyLib.h"

#pragma warning(disable : 4786)

#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <map>
#include <fstream>
#include <conio.h>
#include <ctime>
using namespace std;

using namespace HIT_IR_LTP;	// Important!

ofstream logfile("test_log.txt");
string strText;

int readText();

int test_SplitSentence();
int test_CRFWordSeg();
int test_PosTag();
int test_NER();
int test_Parser();
int test_SRL();

int readText()
{
	ifstream infile("test.txt");
	if ( !infile.is_open() )
	{
		cerr << "can not open: text.txt" << endl;
		exit(0);
	}

	string strLine;
	while ( getline(infile, strLine) )
	{
		strText += strLine;
		strText += "\n";
	}

	infile.close();
	return 0;
}

int test_SplitSentence()
{
	logfile << "\n\n------------"
		<< "\ntest SplitSentence\n";

	// -------------------------
	CreateDOMFromTxt("test.txt");
	
	// Before split sentence, we can get the paragraphs
	logfile << "\n\nbefore split sentence, para info";
	int paraNum = CountParagraphInDocument();
	int i;

	for (i=0; i < paraNum; ++i)
	{
		logfile	<< "\npara " << i << ":";

		const char *para = GetParagraph(i);
		if (para != NULL)
		{
			logfile << "\n" << para;
		}
	}

	SplitSentence();
	SaveDOM("test_splitsentence.xml");

	// -------------------------
	// The xml file has done SplitSentence().
	ClearDOM();		// This is optional, will be done implicitly before CreateDOM.
	CreateDOMFromXml("test_splitsentence.xml");
	
	// Get all sentences in the first way
	logfile << "\n\nsent info, first method";
	int sentNum = CountSentenceInDocument();
	for (i=0; i < sentNum; ++i)
	{
		const char *sent = GetSentence(i);
		if (sent != NULL)
		{
			logfile << "\n" << sent;
		}
	}

	// -------------------------
	CreateDOMFromString(strText.c_str());
	SplitSentence();

	// Get all sentences in the sencond way
	logfile << "\n\nsent info, second method";
	paraNum = CountParagraphInDocument();
	for (i=0; i < paraNum; ++i)
	{
		int sentNum = CountSentenceInParagraph(i);
		for (int j=0; j<sentNum; ++j)
		{
			const char *sent = GetSentence(i, j);
			if (sent != NULL)
			{
				logfile << "\n" << sent;
			}
		}
	}
	SaveDOM("test_splitsentence_string.xml");

	return 0;
}

int test_CRFWordSeg()
{
	logfile << "\n\n------------"
		<< "\ntest CRFWordSeg\n";

	CreateDOMFromTxt("test.txt");
	SplitSentence();
	CRFWordSeg();
	SaveDOM("test_crfwordseg.xml");

	// Get all words in the first way
	logfile << "\n\nwordseg info, first method\n";
	int wordNum = CountWordInDocument();
	for (int i = 0; i < wordNum; ++i)
	{
		const char *word = GetWord(i);
		if (word != NULL)
		{
			logfile << word << " ";
		}
		if ((i+1) % 15 == 0)
		{
			logfile << endl;
		}
	}
	

	// -------------------------
	CreateDOMFromXml("test_crfwordseg.xml");

	// Get all words in the second way
	logfile << "\n\nwordseg info, second method\n";
	int sentNum = CountSentenceInDocument();
	for (int j = 0; j < sentNum; ++j)
	{
		int wordNum = CountWordInSentence(j);
		for (int i = 0; i < wordNum; ++i)
		{
			const char *word = GetWord(j, i);
			if (word != NULL)
			{
				logfile << word << " ";
			}
		}
		logfile << endl;
	}


	// -------------------------
	CreateDOMFromString(strText.c_str());
	SplitSentence();
	CRFWordSeg();

	// Get all words in the third way
	logfile << "\n\nwordseg info, third method\n";
	int paraNum = CountParagraphInDocument();
	for (int k = 0; k < paraNum; ++k)
	{
		int sentNum = CountSentenceInParagraph(k);
		for (int j = 0; j < sentNum; ++j)
		{
			int wordNum = CountWordInSentence(k, j);
			for (int i = 0; i < wordNum; ++i)
			{
				const char* word = GetWord(k, j, i);
				if (word != NULL)
				{
					logfile << word << " ";
				}
			}
			logfile << endl;
		}
	}

	SaveDOM("test_crfwordseg_string.xml");
	return 0;
}

int test_PosTag()
{
	logfile << "\n\n------------"
		<< "\ntest PosTag\n";

	CreateDOMFromTxt("test.txt");
	SplitSentence();
	PosTag();
	SaveDOM("test_postag.xml");

	// Get all words in the first way
	logfile << "\n\npostag info, first method\n";
	int wordNum = CountWordInDocument();
	for (int i = 0; i < wordNum; ++i)
	{
		const char *pos = GetPOS(i);
		if (pos != NULL)
		{
			logfile << pos << " ";
		}
		if ((i+1) % 15 == 0)
		{
			logfile << endl;
		}
	}
	

	// -------------------------
	CreateDOMFromXml("test_postag.xml");

	// Get all words in the second way
	logfile << "\n\npostag info, second method\n";
	int sentNum = CountSentenceInDocument();
	for (int j = 0; j < sentNum; ++j)
	{
		int wordNum = CountWordInSentence(j);
		for (int i = 0; i < wordNum; ++i)
		{
			const char *pos = GetPOS(j, i);
			if (pos != NULL)
			{
				logfile << pos << " ";
			}
		}
		logfile << endl;
	}


	// -------------------------
	CreateDOMFromString(strText.c_str());
	SplitSentence();
	PosTag();

	// Get all words in the third way
	logfile << "\n\npostag info, third method\n";
	int paraNum = CountParagraphInDocument();
	for (int k = 0; k < paraNum; ++k)
	{
		int sentNum = CountSentenceInParagraph(k);
		for (int j = 0; j < sentNum; ++j)
		{
			int wordNum = CountWordInSentence(k, j);
			for (int i = 0; i < wordNum; ++i)
			{
				const char* pos = GetPOS(k, j, i);
				if (pos != NULL)
				{
					logfile << pos << " ";
				}
			}
			logfile << endl;
		}
	}

	SaveDOM("test_crfwordseg_string.xml");
	return 0;
}

int test_NER()
{
	logfile << "\n\n------------"
		<< "\ntest NER\n";

	// -------------------------
	CreateDOMFromTxt("test.txt");
	SplitSentence();
	NER();
	SaveDOM("test_ner.xml");

	// Get all NE in the first way
	logfile << "\n\nNE info, first method\n";
	int wordNum = CountWordInDocument();
	for (int i=0; i < wordNum; ++i)
	{
		const char *ne = GetNE(i);
		if (ne != NULL)
		{
			logfile << ne << " ";
		}
		if ((i+1) % 15 == 0)
		{
			logfile << endl;
		}
	}

	// -------------------------
	CreateDOMFromXml("test_ner.xml");

	// Get all NE in the second way
	logfile << "\n\nNE info, second method\n";
	int sentNum = CountSentenceInDocument();
	for (int j=0; j < sentNum; ++j)
	{
		int wordNum = CountWordInSentence(j);
		for (int i=0; i < wordNum; ++i)
		{
			const char *ne = GetNE(j, i);
			if (ne != NULL)
			{
				logfile << ne << " ";
			}
		}
		logfile << endl;
	}

	// -------------------------
	CreateDOMFromString(strText.c_str());
	SplitSentence();
	NER();

	// Get all NE in the second way
	logfile << "\n\nNE info, third method\n";
	int paraNum = CountParagraphInDocument();
	for (int k=0; k < paraNum; ++k)
	{
		int sentNum = CountSentenceInParagraph(k);
		for (int j=0; j < sentNum; ++j)
		{
			int wordNum = CountWordInSentence(k, j);
			for (int i=0; i < wordNum; ++i)
			{
				const char *ne = GetNE(k, j, i);
				if (ne != NULL)
				{
					logfile << ne << " ";
				}
			}
			logfile << endl;
		}
	}

	SaveDOM("test_ner_string.xml");
	return 0;
}


int test_Parser()
{
	logfile << "\n\n------------"
		<< "\ntest Parser\n";

	// -------------------------
	CreateDOMFromTxt("test.txt");
	SplitSentence();
	GParser();
	SaveDOM("test_parser.xml");

	// Get all Parser in the first way
	logfile << "\n\nParser info, first method\n";
	int wordNum = CountWordInDocument();
	for (int i=0; i < wordNum; ++i)
	{
		pair<int, const char *> parent_relate;
		int ret = GetParse(parent_relate, i);
		if (0 == ret && parent_relate.second != NULL)
		{
			logfile << "<" << parent_relate.first << ", " << parent_relate.second << "> ";
		}
		if ((i+1) % 15 == 0)
		{
			logfile << endl;
		}
	}


	// -------------------------
	CreateDOMFromXml("test_parser.xml");

	// Get all Parser in the second way
	logfile << "\n\nParser info, second method\n";
	int sentNum = CountSentenceInDocument();
	for (int j=0; j < sentNum; ++j)
	{
		int wordNum = CountWordInSentence(j);
		for (int i=0; i < wordNum; ++i)
		{
			pair<int, const char *> parent_relate;
			int ret = GetParse(parent_relate, j, i);
			if (0 == ret && parent_relate.second != NULL)
			{
				logfile << "<" << parent_relate.first << ", " << parent_relate.second << "> ";
			}
		}
		logfile << endl;
	}

	// -------------------------
	CreateDOMFromString(strText);
	SplitSentence();
	GParser();

	// Get all Parser in the third way
	logfile << "\n\nParser info, third method\n";
	int paraNum = CountParagraphInDocument();
	for (int k=0; k < paraNum; ++k)
	{
		int sentNum = CountSentenceInParagraph(k);
		for (int j=0; j < sentNum; ++j)
		{
			int wordNum = CountWordInSentence(k, j);
			for (int i=0; i < wordNum; ++i)
			{
				pair<int, const char *> parent_relate;
				int ret = GetParse(parent_relate, k, j, i);
				if (0 == ret && parent_relate.second != NULL)
				{
					logfile << "<" << parent_relate.first << ", " << parent_relate.second << "> ";
				}
			}
			logfile << endl;
		}
	}

	SaveDOM("test_parser_string.xml");
	return 0;
}


int test_SRL()
{
	logfile << "\n\n------------"
		<< "\ntest SRL\n";

	CreateDOMFromTxt( "test.txt" );
	//CreateDOMFromXml("../../../test_data/test_gb.xml");
	SplitSentence();
	SRL();
	SaveDOM("test_srl.xml");

	// Get all SRL in the first way
	logfile << "\n\nSRL info, first method\n";
	int wordNum = CountWordInDocument();
	for (int i=0; i < wordNum; ++i)
	{
		vector<const char *> vecType;
		vector< pair<int, int> > vecBegEnd;
		
		GetPredArgToWord(vecType, vecBegEnd, i);
		if ( !vecType.empty() )
		{
			logfile << endl;
			for (int m=0; m < vecType.size(); ++m)
			{
				logfile << vecType[m] << " "
					<< vecBegEnd[m].first << " "
					<< vecBegEnd[m].second << " ^ ";
			}
		}
	}


	// -------------------------
	CreateDOMFromXml("test_srl.xml");

	// Get all SRL in the second way
	logfile << "\n\nSRL info, second method\n";
	int sentNum = CountSentenceInDocument();
	for (int j=0; j < sentNum; ++j)
	{
		int wordNum = CountWordInSentence(j);
		for (int i=0; i < wordNum; ++i)
		{
			vector<const char *> vecType;
			vector< pair<int, int> > vecBegEnd;

			GetPredArgToWord(vecType, vecBegEnd, j, i);
			if ( !vecType.empty() )
			{
				logfile << endl;
				for (int m=0; m < vecType.size(); ++m)
				{
					logfile << vecType[m] << " "
						<< vecBegEnd[m].first << " "
						<< vecBegEnd[m].second << " ^ ";
				}
			}
		}
	}

	// -------------------------
	CreateDOMFromString( strText.c_str() ); // [const char *] as parameter, not string, OK
	SplitSentence();
	SRL();

	logfile << "\n\nSRL info, third method\n";
	int paraNum = CountParagraphInDocument();
	for (int k=0; k < paraNum; ++k)
	{
		int sentNum = CountSentenceInParagraph(k);
		for (int j=0; j < sentNum; ++j)
		{
			int wordNum = CountWordInSentence(k, j);
			for (int i=0; i < wordNum; ++i)
			{
				vector<const char *> vecType;
				vector< pair<int, int> > vecBegEnd;

				GetPredArgToWord(vecType, vecBegEnd, k, j, i);
				if ( !vecType.empty() )
				{
					logfile << endl;
					for (int m=0; m < vecType.size(); ++m)
					{
						logfile << vecType[m] << " "
							<< vecBegEnd[m].first << " "
							<< vecBegEnd[m].second << " ^ ";
					}
				}
			}
		}
	}

	SaveDOM("test_srl_string.xml");

	return 0;
}

int main(int argc, char *argv[])
{	
	if (!logfile)
	{
		cerr << "can not open test_log.txt" << endl;
		exit(0);
	}
	readText();

	//test_SplitSentence();
	//test_CRFWordSeg();
	//test_PosTag();
	test_NER();
	//test_Parser();
	//test_SRL();

	logfile.close();
	return 0;
}

