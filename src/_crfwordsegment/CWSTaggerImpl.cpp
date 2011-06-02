#include "CWSTaggerImpl.h"
#include <util/EncodeUtil.h>
#include <util/TextProcess.h>

using namespace std;

namespace CRFPP {

CWSTaggerImpl::CWSTaggerImpl()
{

};

CWSTaggerImpl::~CWSTaggerImpl()
{

};

bool CWSTaggerImpl::add(const string& line)
{
	vector<string> tokens;
	EncodeUtil::split_utf8(line, tokens);

	const char* column[2];
	size_t size=1;
	for(size_t i=0; i<tokens.size(); ++i)
	{
		column[0]=feature_index_->strdup( tokens[i].c_str() );	// only one colume for each token
		if (!add2(size, column, false)) return false;
	}
	return true;
}

bool CWSTaggerImpl::read(std::istream *is)
{
	clear();
	string line;
	int c=0;
	while( (c=is->get()) != -1 )
	{
		if( c==0 || c==' ' || c=='\t' )
		{
			if( line.empty() ) continue;
			else break;
		}
		line.append(1, c);
	}
	if( !line.empty() )
		return add(line);
	return false;
}

const char* CWSTaggerImpl::toString()
{
	os_.assign("");		// clear the buffer
	for(size_t i=0; i<x_.size(); ++i)
	{
		for(vector<const char*>::iterator it=x_[i].begin(); it!=x_[i].end(); ++it)
			os_ << *it << '\t';
		os_ << yname(y(i)) << '\n';
	}
	return const_cast<const char*>(os_.c_str());
}

// equals to
// w1   B
// w2   I
// ...
void CWSTaggerImpl::toString(vector<string> &vec)
{
	size_t size=vec.size();
	vec.resize(size+x_.size());
	int iword=-1;
	for(size_t i=0; i<x_.size(); ++i)
	{
		const char *tag=yname(y(i));
		if( tag!=NULL && x_[i].size()>0 )
		{
			if( tag[0]=='B' )
			{
				iword++;
				vec[size+iword].assign(x_[i][0]);
			}
			else
			{
				if( iword==-1 ) iword++;
				vec[size+iword].append(x_[i][0]);
			}
		}
	}
	vec.resize(size+iword+1);
}

// Notice: parse_stream(...) does not clear the vecWords buffer.
bool CWSTaggerImpl::parse_stream(const string &input, vector<string>& vecWords)
{
	istringstream is(input);
	vecWords.clear();
	while( read(&is) )
	{
		if( !parse() ) break;
		if( x_.empty() ) return true;
		toString(vecWords);
	}
	return !vecWords.empty();
}

bool CWSTaggerImpl::parse_stream(std::istream *is, std::ostream *os)
{
	if( !read(is) || !parse() ) return false;
	if( x_.empty() ) return true;
	toString();
	os->write( os_.data(), (streamsize) os_.size() );
	return true;
}

}; // CRFPP
