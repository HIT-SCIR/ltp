#ifndef __LTP_OPTION__
#define __LTP_OPTION__

#include <string>
#include <iostream>

struct tagNEOption {
    int isEntity;
    int isTime;
    int isNum;

    tagNEOption(int aIsEntity=1, int aIsTime=1, int aIsNum=1) : 
        isEntity(aIsEntity), 
        isTime(aIsTime), 
        isNum(aIsNum) {}

    struct tagNEOption& operator=(const struct tagNEOption & neOpt) {
        isEntity    = neOpt.isEntity;
        isTime      = neOpt.isTime;
        isNum       = neOpt.isNum;
        return *this;
    }

    void output() {
        std::cout << "NE options" << std::endl 
            <<  "isEntity : " << isEntity << std::endl
            <<  "isTime   : " << isTime   << std::endl
            <<  "isNum    : " << isNum    << std::endl;
    }
};

typedef struct tagNEOption NEOption;

struct tagLTPOption {
    std::string strLTPDataPath;
    std::string segmentor_model_path;
    std::string postagger_model_path;
    std::string parser_model_path;
    std::string ner_data_dir;
    std::string srl_data_dir;

    NEOption neOpt;

    void output() {
        std::cout << "ltp options   : " << std::endl
            << "ltp data path : " << strLTPDataPath << std::endl;
        neOpt.output();
    }
};

typedef tagLTPOption LTPOption;

#endif  // end for __LTP_OPTION__
