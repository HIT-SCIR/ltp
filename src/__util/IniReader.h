#pragma once
#ifndef __INIREADER_H__
#define __INIREADER_H__

#include <map>
#include <string>

class IniReader
{
protected:
	std::map<std::string, std::string> _map;	// map for keys and value from INI file
	static const char* null_string;

protected:
	IniReader();

public:
	virtual ~IniReader(void);

	const char* operator() (const std::string& key) const;
	virtual const char* get(const char* key) const;
};

class FileIniReader : public IniReader
{
public:
	FileIniReader(const char *filename);
	~FileIniReader();

//	const char *get(const char* key) const;
};

class StringIniReader : public IniReader
{
public:
	StringIniReader(const char *text);
	~StringIniReader();

//	const char *get(const char* key) const;
};

#endif // __INIREADER_H__
