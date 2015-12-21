//
// Created by jiadongyu on 12/21/15.
//

#ifndef LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H
#define LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H

#define CHECK_RTN_LOGE(x, desc) \
    if(x!=0) { std::cerr<<"error "+desc; return x; }
#define CHECK_RTN_LOGI(x, desc) \
    if(x!=0) { std::cerr<<"info "+desc;  }

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "ltp/ner_dll.h"
#include "ltp/parser_dll.h"
#include "ltp/postag_dll.h"
#include "ltp/segment_dll.h"
#include "ltp/SRL_DLL.h"

class Model {

public:
    const string cws_model_file = "ltp_data/cws.model";
    const string ner_model_file = "ltp_data/ner.model";
    const string par_model_file = "ltp_data/parser.model";
    const string pos_model_file = "ltp_data/pos.model";
    const string srl_model_file = "ltp_data/srl";

    void *cws_model;
    void *ner_model;
    void *par_model;
    void *pos_model;

    Model() {
        int rtn = 0;
        rtn = LodeDefaultModel();
        CHECK_RTN_LOGI(rtn, "error in loading models");
    }
    ~Model(){
        int rtn=0;
        rtn = releaseAll();
        CHECK_RTN_LOGI(rtn," destruct error");
    }

    int LodeDefaultModel() const {
        cws_model = segmentor_create_segmentor(cws_model_file);
        ner_model = ner_create_recognizer(ner_model_file);
        par_model = parser_create_parser(par_model_file);
        pos_model = postagger_create_postagger(pos_model_file);
        if (!cws_model) {
            cerr << "load cws_model error";
            return -1;
        }
        if (!ner_model) {
            cerr << "load ner_model error";
            return -1;
        }
        if (!par_model) {
            cerr << "load par_model error";
            return -1;
        }
        if (!pos_model) {
            cerr << "load pos_model error";
            return -1;
        }
        if (0 != SRL_LoadResource(srl_model_file)) {
            cerr << "load srl_model error";
            return -1;
        }
        cerr << "load succeed!!!" << endl;
        return 0;
    }

    int releaseAll() const {
        int rtn =0;
        rtn = postagger_release_postagger(pos_model);
        CHECK_RTN_LOGE(rtn, "release pos model error");
        rtn = segmentor_release_segmentor(cws_model);
        CHECK_RTN_LOGE(rtn, "release seg model error");
        rtn = parser_release_parser(par_model);
        CHECK_RTN_LOGE(rtn , "release parser model error");
        rtn = ner_release_recognizer(ner_model);
        CHECK_RTN_LOGE(rtn, "release ner model error");
        rtn =  SRL_ReleaseResource();
        CHECK_RTN_LOGE(rtn , "release SRL error");
        return 0;
    }
};


#endif //LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H
