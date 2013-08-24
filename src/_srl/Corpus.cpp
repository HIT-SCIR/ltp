/*
 * File Name     : Corpus.cpp
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 *
 */


#include "Corpus.h"
#include <stdexcept>

using namespace std;

void Corpus::open_corpus(const string &filename) 
{
    m_corpus.close();
    m_corpus.clear();

    m_corpus.open(filename.c_str());
    if (!m_corpus)
    {
        throw runtime_error("Can't open corpus file");
    }
}

bool Corpus::get_next_block(vector<string> &lines)
{
    lines.clear();

    /* if the file has already been read through,
     * return false
     */
    if (m_corpus.eof())
        return false;

    string line;
    while (getline(m_corpus, line))
    {
        if (string::npos == line.find_first_not_of("\t \n"))
        {
            if (lines.size() > 0)
            {
                return true;
            }
        }
        else
        {
            lines.push_back(line);
        }
    }

    if (lines.size() > 0)
    {
        return true;
    }
    else // only blank line
    {
        return false;
    }
}

