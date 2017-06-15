//
// Created by liu on 2017/5/12.
//

#ifndef BILSTM_SRL_WORD_H
#define BILSTM_SRL_WORD_H

#include "iostream"
#include "vector"
#include "Const.h"
#include "Const.h"
using namespace std;

class Word {
  int innerIndex;
  string word;
  string pos;

  int parent;
  string rel;
  string position;

  string predicate;
  vector<string> args;
public:
  Word(int innerIndex,
       const string &word,
       const string &pos,
       int parent,
       const string &rel,
       const string position,
       const string &predicate,
       const vector<string, allocator<string>> &args
    ) : innerIndex(innerIndex), word(word), pos(pos), parent(parent), rel(rel),
        position(position), predicate(predicate), args(args) {}

  Word(int innerIndex,
       const string &word,
       const string &pos,
       int parent,
       const string &rel,
       const string position,
       const string &predicate
  ) : innerIndex(innerIndex), word(word), pos(pos), parent(parent), rel(rel),
      position(position), predicate(predicate) {}

  int getInnerIndex() const {
    return innerIndex;
  }

  const string &getWord() const {
    return word;
  }

  const string &getPos() const {
    return pos;
  }

  const string &getRel() const {
    return rel;
  }

  const string &getPosition() const {
    return position;
  }

  int getParent() const {
    return parent;
  }

  const string &getPredicate() const {
    return predicate;
  }

  void setPredicate(bool isPred) {
    predicate = isPred ? PRED_LABEL : NIL_LABEL;
  }

  bool isPredicate() const {
    return predicate == PRED_LABEL;
  }

  vector<string>& getArgs() {
    return args;
  }

  void setArgs(const vector<string> &args) {
    Word::args = args;
  }
};


#endif //BILSTM_SRL_WORD_H
