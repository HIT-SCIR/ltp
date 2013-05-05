#ifndef __IRNE7TYPERECOG_H__
#define __IRNE7TYPERECOG_H__

#include "crfpp.h"
#include <string>
#include <string.h>
#include <stdlib.h>
using std::string;

/* old version
#include "Model.h"
#include "MyNLPlib.h"
#include "InitDic.h"
#include "RuleNErecog.h"
*/

//using namespace maxent;

/*
 *	实现NE识别的整个过程，包括特征抽取，类别预测，最优搜索
 */

class IRNErecog
{
public:
	IRNErecog();
	~IRNErecog();
	/* old version:
	   void setObject(InitDic *pdic, CModel* model); */
	void crf_set(char *model_path);
	void IRNE7TypeRecog(const string& strSen, string& StrOut, int tagForm, bool* isNEtypeFlag);

/* old version:
private:
	struct SearchNode
	{
		int preNode;  //前一个词对应的nodeNum
		string NEtype;  //NE类别
		double prob;  //累积概率
	};
	typedef struct SearchNode SEARCHNODE; //存放预测搜索的最优结果

	typedef vector<SEARCHNODE> SEARCHVECTOR;

	typedef pair<char, int> FEATURE;

	enum
	{
		SEARCHNODE_NUM = 5,
		TEMPLATE_NUM = 23,
		FormBracket = 1,
		FormBIESO = 2,
		FormMerge = 3
	}; */


/* old version: 
private:

	void NEtaggingNormal(string& NEResult);
	void getNEResult(string& result, int tagform);
	void NEtaggingBasedOnBIOForm1(string& NEResult);
	void NEtaggingBasedOnBIOForm2(string& NEResult);
	void NEtaggingBasedOnBIOForm3(string& NEResult);
	void NEtaggingBasedOnBIESO(string& NEResult);
	//read Template file to vecTemplate
	void readTemplateFile();
	//put words into maps
	void initRuleMaps();    
	void NErecogAtCurrentPostion(int position);
	
	//get proper Word, POS or NEtag
	bool getFeature(const int vecIndex, const int listIndex, 
		const char FeatureChar, string& FeatureOut);

	//combine word, POS, NEtay, and so on, to form a feature 
	void combineOneFeature(int NENODEpos, int preNode, int FeatureNum, string& FeatureOut);
	void seachBestPath(int posVec, int preNode, const vector< pair<string, double> >& vecProb);
	void getBestNEResult();
	void dealFirstPathNode(int posVec, const vector< pair<string, double> >& vecProb);
	void dealOtherPathNode(int posVec, int preNode,
		const vector< pair<string, double> >& vecProb);
	bool isSearchNodeEmpty(int posVec, int Listsize);
	int getListNodeIndexWithSameType(const string& NEtype, int posVec);
	int getNodeIndexWithSmallProb(int posVec);
	int getNodeIndexWithHighProb(int posVec);
	void outvecList(vector<SEARCHVECTOR>& vecOut, ofstream& outfile);
	string getNEtagforHC(const string& Word);
	void cutSingleNE(vector<string>& vecNE);

	void getNEstring(int& senBegpos, int& NEBegpos, string& strOut);
	string getNEPOS(string& NEtype); */

private:
	/* old version:
	vector<SEARCHVECTOR> vecList;  //词的序列
	//vector<FEATURE> vecTemplate[TEMPLATE_NUM];  //特征模板 */

	/* old version: replace next line
	   CModel* Cmodel; */
	CRFPP::Tagger *tagger;

	/* old version 
	InitDic* pdic;
	Ruletest ruletest; */

	/* old version
	//MaxentModel MEmodel;
	vector<string> vecContext;  //存放特征
	vector< pair<string, double> > vecOutcome;  //存放NE预测后的类别和概率信息
	vector< pair<string, string> > vec2paSen;  //存放句子的预处理结果
	vector<string> vecNEResult;


	map<string, int> map_Niend;
	map<string, int> map_Nsend;
	map<string, int> map_Nzend;
	map<string, int> map_sufNh;
	map<string, int> map_preNh;

	bool* bIsNEtypeFlag;
	*/

	/*
	 *	以下for debug
	 */
	//ofstream temp; //for debug
	//ofstream tempProb;
	//ofstream srcProb;
};


#endif
