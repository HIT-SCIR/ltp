///////////////////////////////////////////////////////////////////////
//	CopyRight     : Copyright (c) 2004-2005, IRLab, All rights reserved
//	File Name     : SentenceIterator.h
//	File Summary  : sentence_iterator is a iterator adapter which support
//                  iterate all sentences in a given string. 
//
//	Author        : Robert Chen
//	Create Time   : 2005/4/3
//	Project Name  : Polaris
//	Version       : 1.0
//	Histroy       : 
////////////////////////////////////////////////////////////////////////
#ifndef __POLARIS_SENTENCEITERATOR_H_
#define __POLARIS_SENTENCEITERATOR_H_

#include <string.h>
#include <algorithm>
#include <iterator>
#include <vector>

#include "define.h"
#include "Reader.h"

using namespace std;
using namespace util;

#define NS_DEF(x)  namespace x {
#define NS_END		};

NS_DEF(Chinese)
#define POLARIS_SENTENCE_LENGTH 1024


//translate a chinese character to unsigned short int value
class Character2Int
{
public:
	Character2Int()
	{
		code[2] = '\0';
	}
	unsigned short int operator () (string Separator)
	{
		code[0] = Separator[0];
		code[1] = Separator[1];
		return *(unsigned short int*)code;
	}
private:
	char code[3];
};

//Separator contains a Separator's set, especially, they are unsigned short int value
//the Separator set can be default or be user specified 
//we translate all Separators to int value to speed up the search process
//Here we use sorted vector instead of set, the reason is in <<Effective STL>> : Item 23
class Separator
{
public:
	Separator(const char* fileName)
	{
	/*		if(fileName == NULL)
	{
	cerr << "[Error] Separator::Separator() : You must specify the Separator file in constructor!" << endl;
	return ;
	}
	
	  vector<string> temp;
	  LineReader<string> reader;
	  reader.Read(fileName, std::back_inserter(temp));
	  transform(temp.begin(), temp.end(), std::back_inserter(m_vecSeparator), Character2Int());
		sort(m_vecSeparator.begin(), m_vecSeparator.end());*/
	}
	
	//we use a default Separator set, which containes the normal 
	//Separator that we will use when find sentence
	Separator()
	{
		Character2Int coder = Character2Int();
		m_vecSeparator.push_back(coder(string("¡£")));
		m_vecSeparator.push_back(coder(string("£¡")));
		m_vecSeparator.push_back(coder(string("£¿")));
		m_vecSeparator.push_back(coder(string("£»")));
		m_vecSeparator.push_back(coder(string("£º")));
		//		m_vecSeparator.push_back(coder(string("¡±")));
		m_vecSeparator.push_back(coder(string("¡¡")));
		m_vecSeparator.push_back( '\r' );
		m_vecSeparator.push_back( '\n' );
		m_vecSeparator.push_back( '?' );
		m_vecSeparator.push_back( '!' );
		m_vecSeparator.push_back( ';' );
		
		sort(m_vecSeparator.begin(), m_vecSeparator.end());
	}
	
	bool find(unsigned short int value)
	{
		return binary_search(m_vecSeparator.begin(), m_vecSeparator.end(), value);
	}
	
	bool end()
	{
		return false;
	}
private:
	vector<unsigned short int> m_vecSeparator;
};


class sentence_iterator
{
protected:
	Reader *input;
	TCHAR *_buffer;
	int	  _buf_index, _data_len;	// index pointer for _buffer

	char* _sentence;
	Separator* _separator;

protected:
	void clear()
	{
		if(_sentence != NULL)
			delete[] _sentence;
		_sentence = NULL;
		if(_buffer != NULL)
			delete[] _buffer;
		_buffer = NULL;

		_separator = NULL;
		_buf_index = 0;
		_data_len = 0;
		input = NULL;
	}

public:
	typedef input_iterator_tag iterator_category;
	typedef string value_type;
	typedef int difference_type;
	typedef const char* pointer;
	typedef const char* reference;
	
	sentence_iterator() 
		: _sentence(NULL), _separator(NULL),
		  input(NULL), _buffer(NULL), _buf_index(0), _data_len(0)
	{
	}
	
	explicit sentence_iterator(Reader *in, Separator& sep) 
		: _buf_index(0), _data_len(0), input(in), _buffer(NULL)
	{
		_sentence = NULL;
		_separator = NULL;
		set(in, sep);
	}
	
	~sentence_iterator()
	{
		clear();
	}
	
	void set(Reader *in, Separator& sep)
	{
		clear();
		_buf_index = 0;
		_data_len = 0;
		input = in;
		
		_sentence = new char[POLARIS_SENTENCE_LENGTH+2];
		_buffer = new char[POLARIS_SENTENCE_LENGTH+2];
		_sentence[0] = 0;
		_buffer[0] = 0;
		_separator = &sep;
		
		if(!FindNextSentence())
			clear();
	}
	
	reference operator* ()
	{
		return _sentence;
	}
	
	void operator ++()
	{
		if(!FindNextSentence())
			clear();
	}
	
	sentence_iterator& operator ++(int)
	{
		if(!FindNextSentence())
			clear();
		return *this;
	}
	
	bool operator == (const sentence_iterator& it)
	{
		return (0 == memcmp(this, &it, sizeof(sentence_iterator)));
	}
	
	bool operator != (const sentence_iterator& it)
	{
		return (0 != memcmp(this, &it, sizeof(sentence_iterator)));
	}
	
private:
	bool FindNextSentence()
	{
		char word[3] = {0, 0, 0};	  // for a single character
		
//		int	contentPos(m_ContentPos); //the position we now check
		int sentencePos(0);           //the position into where we should put valid char
		unsigned char c = 0;  //the character we are checking
		unsigned char c1=0;   //the character next to the "c"
		
		while(true)
		{	
			if (_buf_index >= _data_len)
			{
				_data_len = input->read(_buffer, 0, POLARIS_SENTENCE_LENGTH);
				_buf_index = 0;
			}
			if (_data_len <= 0 )
			{
				// input buffer has been empty
				if (sentencePos > 0)
					break;	// yes, we've loaded something
				else
					return false;	// no characters
			}
			else
				c = _buffer[_buf_index++];
			
//			if( c==' ' && sentencePos<1 ) continue;
			
			//recognize a single character, support both Chinese and English
			if(c >= 0x80 || (c1!=0) )
			{
				//				c1 = m_Content[m_ContentPos++];
				if(c1 != 0)
				{
					//store a Chinese character in word
					word[0] = c1;
					word[1] = c;
					
					_sentence[sentencePos++] = c1;
					_sentence[sentencePos++] = c;
					c1 = 0;
				}
				else
				{
					c1 = c;
					continue;
				}
			}
			else
			{
				// store an English character in word
				if( c < ' ' || (c==' ' && sentencePos < 1 ) )
				{
					continue;
				}
				else
				{
					word[0] = c;
					word[1] = 0;
					
					_sentence[sentencePos++] = c;
				}
			}
			
			if( ( _separator->find(*(unsigned short int*)word) != _separator->end() )
				&& sentencePos > 0 )
			{
				//we get a sentence
				_sentence[sentencePos] = '\0';
				sentencePos = 0;
				return true;
			}
			
			//if the sentence is longer than 1024 bytes
			//we think this content is broken and break immediately
			if(c1 == 0 && sentencePos >= POLARIS_SENTENCE_LENGTH-1)
			{
				break;
			}
		}
		_sentence[sentencePos] = 0;
		return true;
	}
	
};

NS_END		// end of Chinese
#endif
