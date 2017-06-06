//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_DATAWORDEMB_H
#define PROJECT_DATAWORDEMB_H

#include <extractor/ExtractorFileToWordEmb.h>
#include "string"
#include "iostream"
#include <unordered_map>
#include <assert.h>
using namespace std;
using namespace extractor;

class WordEmbBuilder {
  unordered_map<string, vector<float>> * emb = NULL;
  bool emb_holding_flag = false;
  unsigned long emb_size = 0;
  vector<float> zero_emb;

  // 禁止拷贝
  WordEmbBuilder &operator=(const WordEmbBuilder &);
public:
  WordEmbBuilder() {}
  WordEmbBuilder(unordered_map<string, vector<float>> & emb) { setEmb(emb); }
  WordEmbBuilder(const string& filename) { loadEmb(filename); }
  ~WordEmbBuilder() {
    if (emb_holding_flag) delete emb;
  }

  void setEmb(unordered_map<string, vector<float>> & emb) {
    assert(WordEmbBuilder::emb == NULL);
    WordEmbBuilder::emb = &emb;
    emb_size = (int) emb.begin()->second.size();
    zero_emb = vector<float>(emb_size, 0);
  }
  void loadEmb(const string& filename) {
    assert(emb == NULL);
    ExtractorFileToWordEmb reader;
    reader.init(filename);
    emb = new unordered_map<string, vector<float>>(reader.run());
    emb_holding_flag = true;
    emb_size = emb->begin()->second.size();
    zero_emb = vector<float>(emb_size, 0);
  }

  const vector<float>& getEmb(const string &key) const {
    assert(emb != NULL);
    if (emb->find(key) != emb->end()) {
      return (*emb)[key];
    } else {
      return zero_emb;
    }
  }


};


#endif //PROJECT_DATAWORDEMB_H
