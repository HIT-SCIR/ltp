#ifndef __LTP_SENTENCE_SPLIT_HPP__
#define __LTP_SENTENCE_SPLIT_HPP__

#include <iostream>
#include <vector>
#include <string>

#include "codecs.hpp"
#include "sentsplit.tab"

namespace ltp {
namespace Chinese {

inline int split_sentence(const std::string & text,
        std::vector<std::string> & sentences,
        int encoding = strutils::codecs::UTF8) {

    sentences.clear();
    std::string sentence; sentence.reserve(512);

    int len = text.size();
    int i = 0;
    int ret = 0;
#define ORD(x) ((unsigned int)((unsigned char)(x)))
    if (encoding == strutils::codecs::UTF8) {
        const int * buff = __double_periods_utf8_buff_x__;
        const int * buff2 = __single_periods_utf8_buff_x__;
        while (i<len) {
            if ((text[i]&0x80)==0) {
                sentence.append(text.substr(i,1));
                if ((text[i]=='\r') || (text[i]=='\n') || 
                        (text[i]=='!') || (text[i]=='?') || (text[i]==';')) {
                    sentences.push_back(sentence);
                    sentence.clear();
                    ++ ret;
                }
                ++ i;
            } else if ((text[i]&0xE0)==0xC0) {
                sentence.append(text.substr(i, 2));
                i += 2;
            } else if ((text[i]&0xF0)==0xE0) {
                unsigned chunk = ((ORD(text[i])<<16)|(ORD(text[i+1])<<8)|ORD(text[i+2]));
                /*std::cerr << text.substr(i,3)<<" "
                    <<chunk<<" "
                    <<(unsigned int)((unsigned char)(text[i]))<<" "
                    <<unsigned(unsigned(text[i])<<16)<<std::endl;*/
                // next character is still a 3-bit char
                if (i+6<len && ((text[i+3]&0xF0)==0xE0)) {
                    int chunk2 = ((ORD(text[i+3])<<16)|(ORD(text[i+4])<<8)|ORD(text[i+5]));
                    if ((chunk==buff[0] && chunk2==buff[1])||
                            (chunk==buff[2] && chunk2==buff[3])||
                            (chunk==buff[4] && chunk2==buff[5])||
                            (chunk==buff[6] && chunk2==buff[7])||
                            (chunk==buff[8] && chunk2==buff[9])||
                            (chunk==buff[10] && chunk2==buff[11])) {
                        sentence.append(text.substr(i,6));
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                        i+=6;
                    } else if ((chunk==buff2[0]) ||
                            (chunk==buff2[1]) ||
                            (chunk==buff2[2]) ||
                            (chunk==buff2[3])) {
                        sentence.append(text.substr(i,3));
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                        i+=3;
                    } else {
                        sentence.append(text.substr(i,3));
                        i+=3;
                    }
                } else {
                    sentence.append(text.substr(i,3));
                    if ((chunk==buff2[0]) ||
                            (chunk==buff2[1]) ||
                            (chunk==buff2[2]) ||
                            (chunk==buff2[3])) {
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                    }
                    i+=3;
                }
            } else if ((text[i]&0xF8)==0xF0) {
                sentence.append(text.substr(i,4));
                i += 4;
            } else {
                std::cerr << "Warning: "
                    << "in sentsplit.hpp split_sentence: string '"
                    << text
                    << "' not encoded in unicode utf-8"
                    << std::endl;
                ++i;
            }
        }
    } else if (encoding == strutils::codecs::GBK) {
        const int * buff = __double_periods_gbk_buff_x__;
        const int * buff2 = __single_periods_gbk_buff_x__;

        while (i<len) {
            if ((text[i]&0x80)==0) {
                sentence.append(text.substr(i,1));
                if ((text[i]=='\r') || (text[i]=='\n') || 
                        (text[i]=='!') || (text[i]=='?') || (text[i]==';')) {
                    sentences.push_back(sentence);
                    sentence.clear();
                    ++ ret;
                }
                ++ i;
            } else {
                int chunk = ((text[i]<<8)|(text[i+1]));
                if (i+4<len) {
                    int chunk2 = ((text[i+2]<<8)|(text[i+3]));
                    if ((chunk==buff[0] && chunk2==buff[1])||
                            (chunk==buff[2] && chunk2==buff[3])||
                            (chunk==buff[4] && chunk2==buff[5])||
                            (chunk==buff[6] && chunk2==buff[7])||
                            (chunk==buff[8] && chunk2==buff[9])||
                            (chunk==buff[10] && chunk2==buff[11])) {
                        sentence.append(text.substr(i,4));
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                        i+=4;
                    } else if ((chunk==buff2[0]) ||
                            (chunk==buff2[1]) ||
                            (chunk==buff2[2]) ||
                            (chunk==buff2[3])) {
                        sentence.append(text.substr(i,2));
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                        i+=2;
                    } else {
                        sentence.append(text.substr(i,4));
                        i+=4;
                    }
                } else {
                    sentence.append(text.substr(i,2));
                    if ((chunk==buff2[0]) ||
                            (chunk==buff2[1]) ||
                            (chunk==buff2[2]) ||
                            (chunk==buff2[3])) {
                        sentences.push_back(sentence);
                        sentence.clear();
                        ++ ret;
                    }
                    i+=2;
                }
            }
        }
    } else {
        return 0;
    }

    if (sentence.size()!=0) {
        sentences.push_back(sentence);
    }
    return ret;
}

}       //  end for namespace Chinese
}       //  end for namespace ltp

#endif  //  end for __LTP_SENTENCE_SPLIT_HPP__
