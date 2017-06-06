//
// Created by jiadongyu on 12/21/15.
//

#ifndef LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H
#define LTP_LANGUAGE_TECHNOLOGY_PLATFORM_GENFEATURE_H_H

#define CHECK_RTN_LOGE(x, desc) \
    if(x!=0) { std::cerr<<"error: "<<desc<<endl; return x; }
#define CHECK_RTN_LOGE_CTN(x, desc) \
    if(x!=0) { std::cerr<<"#####: "<<desc<<endl; continue; }
#define CHECK_RTN_LOGI(x, desc) \
    if(x!=0) { std::cerr<<"info: "<<desc<<endl;  }


#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

#include <vector>
#include <string>
#include <map>
#include <set>

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

    std::vector<std::unordered_set<string>> vec;

    void *cws_model;
    void *ner_model;
    void *par_model;
    void *pos_model;

    const string input = "data/input.txt";
    const string featureOutput = "svm/data/feature.txt";
    const string rawData = "svm/data/raw_data.txt";
    const string saveData ="data/save.txt";


    const bool semantic_tree=true;

    int LoadData(vector<string> &sentences, vector<string> &people,
                 vector<string> &institute, vector<int> &label) const {
        ifstream fin(input);
        sentences.reserve(5000);
        people.reserve(5000);
        institute.reserve(5000);
        label.reserve(5000);
        string tmp;
        while (getline(fin, tmp)) {
            stringstream ss(tmp);
            string s, p, i;
            int l;
            getline(ss, s, '*');
            getline(ss, p, '*');
            getline(ss, i, '*');
            ss >> l;
            if(p.size()==0 || i.size()==0 || s.size()==0){
                return -1;
            }
            sentences.push_back(s);
            people.push_back(p);
            institute.push_back(i);
            label.push_back(l);
            //cerr<<s<<"#"<<p<<"#"<<i<<"#"<<l<<endl;
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

        ner_recognize(ner_model, words, post_tags, nes);

        std::vector<int> heads;
        std::vector<std::string> deprels;

        parser_parse(par_model, words, post_tags, heads, deprels);

        for (int i = 0; i < heads.size(); i++) {
            parseTree.push_back(make_pair(heads[i], deprels[i]));
        }
        //vector<pair<int, vector<pair<string, pair<int, int>>>>> srl;
        //DoSRL(words, post_tags, nes, parseTree, srl);
        //cerr<<"finish srl";
        return 0;
    }

    int main(){
        vector<string> sentences, people, institute;
        vector<int> labels;
        int rtn =0;
        rtn = LoadData(sentences,people,institute,labels);
        CHECK_RTN_LOGE(rtn, "error loading data");
        cerr<<"succ in loading data, size: "<< sentences.size()<<endl;

        ofstream ofs(featureOutput);
        ofstream rawDataWriter(rawData);

        vector<string> toWrite;
        toWrite.resize(sentences.size());


#pragma omp parallel for
        for(int i=0;i<sentences.size();i++){
            int rtn =0;
            cerr<<i<<" start"<<endl;
            string & sentence= sentences[i];
            vector<string> words, post_tags,nes;
            vector<pair<int,string>> parseTree;
            //rtn = parse(sentence,words,post_tags,nes,parseTree);
            cerr<<i<<" parse succ"<<endl;
            CHECK_RTN_LOGE_CTN(rtn,"parse error");
            string feature;
            //save(sentence,people[i],institute[i],labels[i],words,post_tags,nes,parseTree,"data/input/"+to_string(i)+".txt");
            rtn = load(sentence,people[i],institute[i],labels[i],words,post_tags,nes,parseTree,"data/input/"+to_string(i)+".txt");
            CHECK_RTN_LOGE_CTN(rtn, "error loading");

            rtn = getFeature(sentence, people[i],institute[i],labels[i],words,post_tags,nes,parseTree,feature);
            CHECK_RTN_LOGE_CTN(rtn, "error getting feature");
            cerr<<"succ in get feature: "<<feature<<endl<<endl;
            string tmp;
            if(labels[i]>0){
                tmp = "1 ";
            }else{
                tmp ="-1 ";
            }
            //labelData(words,nes,toWrite[i]);
            toWrite[i]=tmp + feature;
        }

        for(int i=0;i<toWrite.size();i++){
            if(toWrite[i].size()>0){
                ofs<< toWrite[i]<<endl;
                rawDataWriter<< sentences[i]<<"*"<<people[i]<<"*"<<institute[i]<<"*"<<labels[i]<<endl;
            }
        }
        ofs.close();
        return 0;
    }

    void save(const string & sentence, const string &person, const string &institute,
             const int &label, const vector<string> &words,
             const vector<string> &post_tags, const vector<string> &nes,
             const vector<pair<int, string>> &parseTree,const string file){
        ofstream of(file);
        of<<sentence<<endl;
        of<<person<<endl;
        of<<institute<<endl;
        of<<label<<endl;
        of<<words.size()<<endl;
        for(int i=0;i<words.size();i++){
            of<<words[i]<<" ";
        }
        of<<endl;

        of<<post_tags.size()<<endl;
        for(int i=0;i<post_tags.size();i++){
            of<<post_tags[i]<<" ";
        }
        of<<endl;

        of<<nes.size()<<endl;
        for(int i=0;i<nes.size();i++){
            of<<nes[i]<<" ";
        }
        of<<endl;

        of<<parseTree.size()<<endl;
        for(int i=0;i<parseTree.size();i++){
            of<<parseTree[i].first<<" "<<parseTree[i].second<<" ";
        }
        of<<endl;
    }

    int load( string & sentence,  string &person,  string &institute,
              int &label,  vector<string> &words,
              vector<string> &post_tags,  vector<string> &nes,
              vector<pair<int, string>> &parseTree , const string  file){
        ifstream ifs(file);
        if(!ifs.good()){
            return -1;
        }
        ifs>>sentence;
        ifs>>person;
        ifs>>institute;
        ifs>>label;
        int tmp;
        ifs>>tmp;
        words.clear();
        words.resize(tmp);
        for(int i=0;i<tmp;i++){
            ifs>>words[i];
        }

        ifs>>tmp;
        post_tags.clear();
        post_tags.resize(tmp);
        for(int i=0;i<tmp;i++){
            ifs>>post_tags[i];
        }
        ifs>>tmp;
        nes.clear();
        nes.resize(tmp);
        for(int i=0;i<tmp;i++){
            ifs>>nes[i];
        }

        ifs>>tmp;
        parseTree.clear();
        parseTree.resize(tmp);
        for(int i=0;i<tmp;i++){
            ifs>>parseTree[i].first>>parseTree[i].second;
        }
        return 0;
    }

    int labelData(const vector<string> & words, const vector<string> & nes, string & out){
        out.clear();
        unordered_map<string, string> m;
        m["Nh"]=" /nr";
        m["Ni"]=" /nt";
        m["Ns"]=" /ns";

        for(int i=0;i<words.size();i++){
            if(nes[i]=="S-Ns" || nes[i]=="S-Nh" ||  nes[i]=="S-Ni" || nes[i]=="B-Ns" ||nes[i]=="B-Nh"||nes[i]=="B-Ni" ){
                string tag = nes[i].substr(2);
                out.push_back('{');
               // out.append(words[i]);
            }
            //if(nes[i].at(0)=='O' || nes[i].at(0)=='I' || nes[i].at(0)=='E'){
                out.append(words[i]);
            //}
            if(nes[i].at(0)=='S' || nes[i].at(0)=='E'){
                string tag = nes[i].substr(2);
                //out.append(words[i]);
                out.append(m[tag]);
                out.push_back('}');
            }
        }
        //out.push_back('\n');
    }



    int getFeature(const string & sentence, const string &person, const string &institute,
                   const int &label, const vector<string> &words,
                   const vector<string> &post_tags, const vector<string> &nes,
                   const vector<pair<int, string>> &parseTree, string & feature
    ) const {
        int dePerson=-1,deInstitute=-1;
        int rtn=0;

        set<int> PosP,PosI;
        rtn = getDetectedPI(person,institute,words,nes,dePerson,deInstitute,PosP,PosI);
        CHECK_RTN_LOGE(rtn,"detect P,i error");
        cerr<<"ins,per: "<<deInstitute<< " "<<dePerson;
        int subRoot=-1;
        rtn = getRoot(dePerson,deInstitute,parseTree,subRoot);
        CHECK_RTN_LOGE(rtn, "get tree root error");
        cerr<<"root: "<<subRoot<<" :"<<words[subRoot];
        vector<vector<int> > children;
        int treeRoot=-1;
        rtn = getChildren(parseTree,children,treeRoot);
        CHECK_RTN_LOGE(rtn, "error in getting children");
        cerr<<" children size:"<<children.size()<<" parseTree"<< parseTree.size()<<endl;
        string fullTree;
        rtn = getTree(post_tags, parseTree,words,children,subRoot,fullTree);
        CHECK_RTN_LOGE(rtn, "error in getting tree");
        string simpleTree;
        rtn = getSimpleTreeWrapper(post_tags, parseTree,words,children,subRoot,simpleTree,dePerson,deInstitute);
        CHECK_RTN_LOGE(rtn, "error in getting simple tree");

        string tree = "|BT| " +fullTree +" |BT| " + simpleTree +" |ET| ";

        string vc;
        rtn =getFeatureVec(sentence,post_tags,words,vc,dePerson,deInstitute,PosP,PosI);
        string vec ="|BV|"+vc+"|EV|";
        feature = tree +vec;
        return 0;
    }
private:
    int getTree(const vector<string> & post_tags, const vector<pair<int, string>> &parseTree, const vector<string> & words,
                const vector<vector<int>> &children,
                const int root, string & feature) const {
        feature.clear();
        string subTree;
        //cerr<<"getting root:"<<root<<" :size:"<<children[root].size()<<" @ " <<endl;
        int rtn = 0;
        if(children[root].size()==0) {
            if(semantic_tree){
                feature = "("+parseTree[root].second+" "+words[root]+")";
            } else {
                feature = "("+parseTree[root].second+" "+post_tags[root]+")";
            }

            return 0;
        }
        feature="(" +parseTree[root].second;
        for(int i=0;i<children[root].size();i++) {
            rtn = getTree(post_tags, parseTree,words, children, children[root][i], subTree);
            feature.push_back(' ');
            feature.append(subTree);
        }
        feature.push_back(')');
        //cerr<<"finish get Tree: "<<feature<<endl;
        return 0;
    }

    int getSimpleTreeWrapper(const vector<string> & post_tags, const vector<pair<int, string>> &parseTree,
                             const vector<string> & words,
                             const vector<vector<int>> &children,
                             const int root, string & feature, const int p, const int i) const{
        int rtn =0;
        if(root==p){
            rtn = getSimplePath(post_tags, parseTree, words,children,root,i,feature);
            CHECK_RTN_LOGE(rtn,"root=p,getSimplePath error");
        }else if(root==i){
            rtn = getSimplePath(post_tags, parseTree,words,children,root,p,feature);
            CHECK_RTN_LOGE(rtn,"root=i,getSimplePath error");
        }else{
            rtn =getSimpleTree(post_tags, parseTree,words,children,root,feature,p,i);
            CHECK_RTN_LOGE(rtn,"i,p,getSimpleTree error");
        }
        return 0;
    }

    int getSimplePath(const vector<string> & post_tags, const vector<pair<int, string>> &parseTree,
                      const vector<string> & words,
                      const vector<vector<int>> &children,
                      const int root,  int dest, string & feature) const {
        feature.clear();
        string subTree;
       // cerr<<"getting root:"<<root<<" :size:"<<children[root].size()<<" @ " <<endl;
        int rtn = 0;

        if(semantic_tree){
            feature = "(" + parseTree[dest].second + " " + words[dest] + ")";
        }else{
            feature = "(" + parseTree[dest].second + " " + post_tags[dest] + ")";
        }


        while(parseTree[dest].first-1>=0){

            if(parseTree[dest].first-1==root){
                break;
            }else{
                //feature="("+parseTree[dest].second+" "+feature+")";
                feature="("+parseTree[dest].second+" "+feature+")";
            }
            dest = parseTree[dest].first-1;
        }

        //cerr<<"finish get Tree: "<<feature<<endl;
        return 0;
    }

    int getSimpleTree(const vector<string> & post_tags, const vector<pair<int, string>> &parseTree,
                      const vector<string> & words,
                      const vector<vector<int>> &children,
                      const int root, string & feature, const int p, const int i)const{
        feature.clear();
        string subTree;
        //cerr<<"getting root:"<<root<<" :size:"<<children[root].size()<<" @ " <<endl;
        int rtn = 0;


        if( root==p || root==i ) {
            if(!semantic_tree) {
                feature = "(" + parseTree[root].second + " " + post_tags[root] + ")";
            }else{
                feature = "(" + parseTree[root].second + " " + words[root] + ")";
            }
            return 0;
        }

        string leaf ="";
        for(int i=0;i<children[root].size();i++) {
            int subRoot= children[root][i];
            rtn = getSimpleTree(post_tags, parseTree,words, children, subRoot, subTree,p,i );
            if(subTree.size()>0) {
                leaf.append(subTree);
                leaf.push_back(' ');
            }
        }
        if(leaf.size()>0){
            feature= "(" +parseTree[root].second + leaf+")";
            //feature= "(" +words[root] + leaf+")";
        }

        //cerr<<"finish get Tree: "<<feature<<endl;
        return 0;

    }

    int getChildren(const vector<pair<int, string>> &parseTree, vector<vector<int>> & children, int & root) const{
        children.clear();
        children.resize(parseTree.size());
        for(int i=0;i<parseTree.size();i++){
            if(parseTree[i].first-1<0){
                root=i;
            }else {
                children[parseTree[i].first-1].push_back(i);
            }
        }
        //cerr<<"finish get children"<<endl;
        return 0;
    }

    int getRoot(int a, int b, const vector<pair<int,string> > & parseTree , int & root) const {

       // for(int i=0;i<parseTree.size();i++){
       //     cerr<<parseTree[i].first<<" "<<parseTree[i].second<<endl;
       // }
        if(a==b || a<0 || b<0){
            cerr<<" a b not valid"<<a << b<<endl;
            return -1;
        }
        unordered_set<int> path;
        path.insert(a);
        while(parseTree[a].first-1>=0){
            a=parseTree[a].first-1;
            path.insert(a);
        }
        while(parseTree[b].first-1>0){
            b=parseTree[b].first-1;
            if(path.count(b)>0){
                root = b;
                break;
            }
        }
        if(root<0){
            cerr<<" root is -1"<<endl;
            return -1;
        }
        //cerr<<"finish get root: "<<endl;
        return 0;
    }

    int getDetectedPI(const string &person, const string &institute,
                      const vector<string> &words, const vector<string> &nes,
                      int &dePerson, int &deInstitute, set<int> & PosP, set<int> & PosI ) const {

        map<string,int> dPs, dIs;
        string tI,tP;
        for(int i=0;i<words.size();i++){
            if(nes[i].find("O")!=std::string::npos){
                if(tI.size()>0){
                    if(dIs.count(tI)>0) {
                        cerr << "more than one ins " << tI;
                        return -1;
                    }
                    dIs[tI]=i-1;
                    tI.clear();
                }
                if(tP.size()>0){
                    if(dPs.count(tP)>0){
                        cerr<< "more than one people "<< tP<< endl;
                        return -1;
                    }
                    dPs[tP]=i-1;
                    tP.clear();
                }
            }
            else if(nes[i].find("Nh")!=std::string::npos){
                tP.append(words[i]);
            }else if(nes[i].find("Ni")!=std::string::npos){
                tI.append(words[i]);
            }
            //cerr<<nes[i];
           // cerr<<i<<" "<<words[i]<<" "<<nes[i]<<endl;
        }

        if(tI.size()>0){
            if(dIs.count(tI)>0) {
                cerr << "more than one ins " << tI;
                return -1;
            }
            dIs[tI]=(int)words.size()-1;
            tI.clear();
        }
        if(tP.size()>0){
            if(dPs.count(tP)>0){
                cerr<< "more than one people "<< tP;
                return -1;
            }
            dPs[tP]=(int)words.size()-1;
            tP.clear();
        }

        for(auto a: dPs){
            PosP.insert(a.second);
            cerr<<"*"<<a.first<<endl;
            if(a.first.find(person)!=std::string::npos || person.find(a.first)!=std::string::npos){
                dePerson=a.second;
            }
        }
        for(auto a:dIs){
            PosI.insert(a.second);
            int m=100;
            cerr<<"*"<<a.first<<endl;
            if(a.first.find(institute)!=std::string::npos ){
                deInstitute=a.second;
            }
            if(institute.find(a.first)!=std::string::npos ){
                if((int)institute.size()-(int)a.first.size()<m){
                    deInstitute=a.second;
                    m=(int)institute.size()-(int)a.first.size();
                }
            }
        }
        //<<"finish detect PI"<<endl;
        return 0;
    }

    int getFeatureVec(const string & sentence,
                      const vector<string> & post_tags,
                      const vector<string> & words,
                      string & feature, const int p, const int i, const set<int> & PosP,const set<int> & PosI ) const {
        vector<int> fe;
        fe.resize(3*vec.size()+30);
        int k=0;
        /*
        for(int j=0;j<vec.size();j++){
            int idx=k*vec.size()+j;
            fe[idx]=has(sentence,vec[j]);
        }
         */
        k=1;
        string sen;
        //cerr<<"from "<< min(p,i)<<" to "<<max(p,i)<<endl;
        for(int j=min(p,i);j<=max(p,i);j++){
            sen.append(words[j]);
        }
        for(int j=0;j<vec.size();j++){
            int idx=k*vec.size()+j;
            fe[idx]=has(sen,vec[j]);
        }
        k=2;
        sen.clear();
        for(int j=max(min(p,i)-3,0);j<=min(max(p,i)+3,(int)words.size()-1);j++){
            sen.append(words[j]);
        }
        for(int j=0;j<vec.size();j++){
            int idx=k*vec.size()+j;
            fe[idx]=has(sen,vec[j]);
        }
        int base = 3* vec.size();

        fe[++base]= p>i?1:0; // 7

        int mi=min(p,i);
        int mx=max(p,i);

        fe[++base]=0; //9
        for(auto a:PosI){
            if(a>mi && a< mx){
                fe[base]=1;
                break;
            }
        }

        fe[++base]=0; //10
        for(auto a:PosP){
            if(a>mi && a< mx){
                fe[base]=1;
                break;
            }
        }

        //fe[++base]=p-i;//11

        map<string, int > mpTag,mpWords;
        for(int j=min(p,i);j<=max(p,i);j++){
            mpTag[post_tags[j]]++;
            mpWords[words[j]]++;
        }
        vector<string> testTag; testTag.push_back("v"); //testTag.push_back("wp");
        vector<string> wordTag; wordTag.push_back("，"); //wordTag.push_back("（");wordTag.push_back("）");
        for(auto a:testTag){
            fe[++base]=mpTag.count(a)>0?1:0;
        }
        for(auto a:wordTag){
            fe[++base]=mpWords.count(a)>0?1:0;
        }

        feature.clear();
        feature.reserve(100);
        for(int j=0;j<fe.size();j++){
            if(fe[j]!=0) {
                feature.append(to_string(j+1));
                feature.push_back(':');
                feature.append(to_string(fe[j]));
                feature.push_back(' ');
            }
        }
        return 0;
    }
    inline int  has(const string & sentence, const unordered_set<string> & st) const {
        for(auto a: st){
            if(sentence.find(a)!=std::string::npos){
                //cerr<<sentence<< " find :"<<a << endl;
                return 1;
            }
        }
        return 0;
    }


public:
    Model() {
        int rtn = 0;
        rtn = LodeDefaultModel();
        CHECK_RTN_LOGI(rtn, "error in loading models");
        rtn = LoadKeyWords();
        CHECK_RTN_LOGI(rtn, "error in loading keywords");
    }

    ~Model() {
        int rtn = 0;
        rtn = releaseAll();
        CHECK_RTN_LOGI(rtn, " destruct error");
    }

    int LoadKeyWords(){
        int fileListNum=2;
        vec.resize(fileListNum);
        for(int i=0;i<fileListNum;i++){
            ifstream fin("data/"+to_string(i)+".txt");
            string tmp;
            if(!fin.good()){
                return -1;
            }
            while(fin>>tmp){
                vec[i].insert(tmp);
            }
            fin.close();
        }
        return 0;
    }

    int LodeDefaultModel() {
        cws_model = segmentor_create_segmentor(cws_model_file.c_str());
        ner_model = ner_create_recognizer(ner_model_file.c_str());
        par_model = parser_create_parser(par_model_file.c_str());
        pos_model = postagger_create_postagger(pos_model_file.c_str());
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
