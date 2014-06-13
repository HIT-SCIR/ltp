#ifndef __LTP_PARSER_DICT_COLLECTIONS_H__
#define __LTP_PARSER_DICT_COLLECTIONS_H__

#include <iostream>
#include <vector>
#include "utils/stringmap.hpp"
#include "utils/smartmap.hpp"
#include "parser/instance.h"

namespace ltp {
namespace parser {

using namespace std;
using namespace ltp::utility;


// declariation of dictionary, this is specially needed
// by the observer design pattern
class Dictionary;

// class of a collection of dictionary
// a index counter is shared within several dictionary.
class DictionaryCollections {
public:
  DictionaryCollections(int num_dicts);
  ~DictionaryCollections();

  /*
   * Dump the dictionary collections into output stream
   *
   *  @param[out]   out   the output stream
   */
  void dump(ostream & out);

  /*
   * Load the dictionary collections from input stream,
   * return true if dictionary successfully loaded, otherwise
   * false.
   *
   *  @param[in]    in    the input stream
   *  @return     bool  true on success, otherwise false.
   */
  bool load(istream & in);

  /*
   * Get the size of dictionary collections
   *
   *  @return     size_t  the size of the dictionary
   */
  size_t dim() const;

  /*
   * Retrieve the certain key in one of the dictionaries in this
   * collection. If create is specified, this key is created on
   * the condition that it is not in the dictionary. Return the 
   * index of the key, -1 on failure
   *
   *  @param[in]  tid   the index of the dictionary
   *  @param[in]  key   the key
   *  @param[in]  create  insert the key to dictionary if create
   *            if true.
   *  @return   int   the index of the key, -1 on failure.
   */
  int retrieve(int tid, const char * key, bool create);

  /*
   * Get the ith Dictionary
   *
   *  @param[in]  i         the index of the dictionary
   *  @return   Dictionary *  the dictionary
   */
  Dictionary * getDictionary(int i);

  /*
   * Get size of dicts
   *
   *  @return   int       the size of the dictionary
   */
  int size();

public:
  int idx;    /*< the shared index among dictionaries */

private:
  vector<Dictionary *> dicts;
};

// the dictionary class
// it's wrapper of class SmartMap<int>
class Dictionary {
public:
  Dictionary(DictionaryCollections * coll): 
    collections(coll) {}

  //StringMap<int>      database;
  SmartMap<int>       database;
  DictionaryCollections * collections;

  inline int retrieve(const char * key, bool create) {
    int val;

    if (database.get(key, val)) {
      return val;
    } else {
      if (create) {
        val = collections->idx;
        database.set(key, val);
        // database.unsafe_set(key, val);
        ++ collections->idx;
        return val;
      }
    }

    return -1;
  }

  inline int size() {
    return database.size();
  }
};

// labelcollections is a bi-direction map.
// it support two way of retrieving
//
//  * string key -> int index
//  * int index -> string key
//
}     //  end for namespace parser
}     //  end for namespace ltp
#endif  //  end for __LTP_PARSER_DICT_COLLECTIONS_H__
