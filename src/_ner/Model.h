#ifndef IR_CModel_H
#define IR_CModel_H

#include "maxentmodel.hpp"
//#include "../__maxent/maxentmodel.hpp"
//#define STL_USING_ALL

// #include <stl.h>
#include <map>

using namespace maxent;


class CModel
{
public:
	void LoadMEModel(const string& path);
	void ReleaseNEModle();
private:
	typedef pair<char, int> FEATURE;
	enum
	{
		SEARCHNODE_NUM = 5,
		TEMPLATE_NUM = 23
	};
	void readTemplateFile(const string& path);
public:
	MaxentModel MEmodel;	
	vector<FEATURE> vecTemplate[TEMPLATE_NUM];
};
#endif;
