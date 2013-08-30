/*
 * File Name     : Corpus.h
 * Author        : msmouse
 *
 * Updated by    : jiangfeng
 * Update Time   : 2013-08-21
 *
 */


#ifndef _CORPUS_H_
#define _CORPUS_H_

#include <vector>
#include <string>
#include <fstream>

class Corpus
{
public:
    Corpus() {}

    /* new Corpus corresponding to file "filename" */
    explicit Corpus(const std::string &filename) {open_corpus(filename);}

    ~Corpus() {}

    /* open a corpus file for input */
    void open_corpus(const std::string &filename);

    /* get the next block, blocks are separated with a blank line */
    bool get_next_block(std::vector<std::string> &lines);

private:
    std::ifstream m_corpus;
};

#endif

