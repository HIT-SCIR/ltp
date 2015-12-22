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
#include <ifstream>
#include <ofstream>
#include <sstream>

#include <vector>
#include <string>
#include <map>

#include "ltp/ner_dll.h"
#include "ltp/parser_dll.h"
#include "ltp/postag_dll.h"
#include "ltp/segment_dll.h"
#include "ltp/SRL_DLL.h"

using namespace std;

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

    const string input = "../data/input.txt";
    const string featureOutput = "../svm/data/feature.txt";

    int LoadData(vector<string> &sentences, vector<string> &people,
                 vector<string> &institute, vector<int> &label) const {
        ifstream fin(input);
        string tmp;
        while (getline(fin, tmp)) {
            stringstream ss(tmp);
            string s, p, i;
            int l;
            getline(ss, s, '*');
            getline(ss, p, '*');
            getline(ss, i, '*');
            ss >> l;
        }
        return 0;
    }

    int parse(const string &sentence, vector<string> &words,
              vector<string> &post_tags, vector<string> &nes,
              vector<pair<int, string>> &parseTree
    ) const {
        words.clear();
        post_tags.clear();
        nes.clear();
        parseTree.clear();
        int len = segmentor_segment(cws_model, sentence, words);
        postagger_postag(pos_model, words, post_tags);
        ner_recognize(ner_model, words, postags, nes);
        std::vector<int> heads;
        std::vector<std::string> deprels;

        parser_parse(par_model, words, post_tags, heads, deprels);

        for (int i = 0; i < heads.size(); i++) {
            parseTree.push_back(make_pair(heads[i], deprels[i]));
        }
        vector<pair<int, vector<pair<string, pair<int, int>>>>> srl;
        DoSRL(words, post_tags, nes, parseTree, srl);
        return 0;
    }

    int getFeature(const string &person, const string &institute,
                   const int &label, const vector<string> &words,
                   const vector<string> &post_tags, const vector<string> &nes,
                   const vector<pair<int, string>> &parseTree, string feature
    ) const {

    }

    int getDetectedPI(const string &person, const string &institute,
                      const vector<string> &words, const vector<string> &nes,
                      string &dePerson, string &deInstitute) const {
        dePerson.clear();deInstitute.clear();
        set<string > dPs, dIs;
        string tP,tI;
        for(int i=0;i<words.size();i++){
            if(nes[i].find("O")>0){
                if(tI.size()>0){
                    dIs.insert(tI);
                    tI.clear();
                }
                if(tP.size()>0){
                    dPs.insert(tP);
                    tP.clear();
                }
            }
            else if(nes[i].find("Nh")>0){
                tP.append(words[i]);
            }else if(nes[i].find("Ni")>0){
                tI.append(words[i]);
            }
        }

        if(tI.size()>0){
            dIs.insert(tI);
            tI.clear();
        }
        if(tP.size()>0){
            dPs.insert(tP);
            tP.clear();
        }

        for(auto a: dPs){
            if(a.find(person)>0 || person.find(a)>0){
                dePerson=a;
            }
        }
        for(auto a:dIs){
            if(a.find(institute)>0 || institute.find(a)>0){
                deInstitute=a;
            }
        }
        
    }

    Model() {
        int rtn = 0;
        rtn = LodeDefaultModel();
        CHECK_RTN_LOGI(rtn, "error in loading models");
    }

    ~Model() {
        int rtn = 0;
        rtn = releaseAll();
        CHECK_RTN_LOGI(rtn, " destruct error");
    }


    int LodeDefaultModel() {
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

    int releaseAll() {
        int rtn = 0;
        rtn = postagger_release_postagger(pos_model);
        CHECK_RTN_LOGE(rtn, "release pos model error");
        rtn = segmentor_release_segmentor(cws_model);
        CHECK_RTN_LOGE(rtn, "release seg model error");
        rtn = parser_release_parser(par_model);
        CHECK_RTN_LOGE(rtn, "release parser model error");
        rtn = ner_release_recognizer(ner_model);
        CHECK_RTN_LOGE(rtn, "release ner model error");
        rtn = SRL_ReleaseResource();
        CHECK_RTN_LOGE(rtn, "release SRL error");
        return 0;
    }
};


#endif //LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H
