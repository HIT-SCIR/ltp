#include "Model.h"
#include "MyNLPlib.h"

//CModel::CModel()
//{
	//totalWordNum = 0;
//}

//CModel::~CModel()
//{
	//DestoryData();
//}
void CModel::ReleaseNEModle()
{

}
void CModel::LoadMEModel(const string& path)
{
	readTemplateFile(path);
	//initRuleMaps();

	string model = path + "/MEirBIO";
	MEmodel.load(model);
	cout << "MEmodel load over" << endl;

	/*temp.open("outNE.txt");
	tempProb.open("outProb.txt");
	srcProb.open("srcProb.txt");*/
}

void CModel::readTemplateFile(const string &path)
{
	ifstream inTemp((path + "/Template.dic").c_str());
	if (!inTemp)
	{
		cerr << "Can not open Template file!" << endl;
		exit(-1);
	}

	string strFeature;
	FEATURE featurer;
	string offset;
	int LineNum = 0;

	vector<string> vecTemp;

	while (getline(inTemp, strFeature))
	{
		vecTemp.clear();
		splitSenByChar(strFeature, '/', vecTemp);

		for (int i=0; i<(int)vecTemp.size(); i++)
		{
			featurer.first = vecTemp[i].at(0);
			if (vecTemp[i].size() == 1)
			{
				featurer.second = 0;
			}
			else
			{
				offset = vecTemp[i].substr(1);
				featurer.second = atoi(offset.c_str());
			}            
			vecTemplate[LineNum].push_back(featurer);
		}

		LineNum++;						
	}

	inTemp.close();
}
