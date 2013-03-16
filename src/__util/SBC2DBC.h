#ifndef __SBC2DBC_H__
#define __SBC2DBC_H__

#include <string>
#include <map>

#pragma once
using namespace std;

class SBC2DBC
{
public:
	SBC2DBC();
	~SBC2DBC();
	void DoSBC2DBC(const std::string &str, std::string &strResult);
	void DoSBC2DBC_if_begin_with_SBC(const std::string &str, std::string &strResult);

private:
	void Initialize();

private:
	std::map<std::string, char> m_mapSBC2DBC;
		static const int SBC_TABLE_SIZE = 69;
};

#endif

