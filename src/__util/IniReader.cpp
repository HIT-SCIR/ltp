#include "IniReader.h"
#include <fstream>
#include <string.h>

using namespace std;

const char* IniReader::null_string = "";
//////////////////////////////////////////////////////////////////////////

IniReader::IniReader()
{
}

IniReader::~IniReader(void)
{
}

const char* IniReader::operator() (const string& key) const
{
	map<string, string>::const_iterator i;
	if( (i=_map.find(key)) == _map.end() )
		return null_string;	// empty object
	return (i->second).c_str();
}

const char* IniReader::get(const char* key) const
{
	map<string, string>::const_iterator i;
	string k(key);
	if( (i=_map.find(k)) == _map.end() )
		return null_string;	// empty object
	return (i->second).c_str();
}

//////////////////////////////////////////////////////////////////////////

FileIniReader::FileIniReader(const char *filename)
{
	ifstream in(filename);
	if(!in)
		throw std::string("Cannot open ") + string(filename);

	const int bufsize = 1000;
	char line[bufsize], *p;
	string key;
	while( in.getline(line, bufsize) )
	{
		p = strtok(line, "=");
		if( p == NULL ) continue;
		if( *p == '#' ) continue;	// eliminate the comments

		// extract 'key'
		size_t len = strlen(p);
		while( p[len-1] <=' ' ) len--;
		p[len] = 0;
		key = p;

		// extract 'value'
		p = strtok(NULL, "=");
		if( p == NULL ) continue;
		// triming leading spaces
		while( *p!=0 && *p<=' ' ) p++;
		// triming tailing spaces
		len = strlen(p);
		while( p[len-1] <=' ' ) len--;
		p[len] = 0;

		_map[key] = p;
	}
	in.close();
	return;
}

FileIniReader::~FileIniReader()
{

}

//////////////////////////////////////////////////////////////////////////

StringIniReader::StringIniReader(const char *text)
{

}

StringIniReader::~StringIniReader()
{

}
