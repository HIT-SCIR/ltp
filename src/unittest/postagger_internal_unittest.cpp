#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "postagger/model.h"
#include "postagger/constrainutil.hpp"
#include "utils/smartmap.hpp"
#include "utils/tinybitset.hpp"

using namespace std;
using namespace ltp::utility;

const int kMaxTagIndex = 27;
const char * kTags[] = {  "a",  "b",  "c",
  "d",  "e",  "h",  "i",  "j",  "k",  "m",
  "n",  "nd", "nh", "ni", "nl", "ns", "nt",
  "nz", "o",  "p",  "q",  "r",  "u",  "v",
  "wp", "ws", "z"};

TEST(postagger_unittest, test_tags_of_model) {
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());

  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  EXPECT_EQ(kMaxTagIndex, model->labels.size());

  for (int i = 0; i < kMaxTagIndex; ++ i) {
    EXPECT_GE(model->labels.index(kTags[i]), 0);
    EXPECT_LT(model->labels.index(kTags[i]), kMaxTagIndex);
  }

  delete model;
}

TEST(postagger_unittest, test_load_constrain) {
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-single.constrain");
  EXPECT_EQ(nr_constraints, 1);
}

TEST(postagger_unittest, test_load_constrain_to_uninitialized_model) {
  ltp::postagger::Model * model = new ltp::postagger::Model();
  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-single.constrain");
  EXPECT_EQ(nr_constraints, 0);
}

TEST(postagger_unittest, test_load_constrain_with_unknown_tag) {
  // It's expected to show a warning log complain that the tag
  // was not known.
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-unknown.constrain");
  EXPECT_EQ(nr_constraints, 0);
}

TEST(postagger_unittest, test_load_constrain_with_known_and_unknown_tag) {
  // There are two tag in the constrain file (one is known and another one
  // unknown). One constrain should be loaded, but only one bit should be
  // activated.
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-known-unknown.constrain");
  EXPECT_EQ(nr_constraints, 1);
}

TEST(postagger_unittest, test_load_constrain_function_correction) {
  
  const std::string model_file = "./ltp_data/pos.model";
  const std::string constraints_file = "./test_data/unittest/postag-known.constrain";

  std::ifstream mfs(model_file.c_str());
  std::ifstream lfs(constraints_file.c_str());

  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);
  load_constrain(model, constraints_file.c_str());
  
  typedef std::vector< int > int_vec;
  typedef std::vector< std::string > str_vec;
  typedef std::map< std::string, str_vec > Map;
  typedef Map::iterator MapIt;
  typedef Map::const_iterator MapConstIt;

  Map mapping_stl;
  Map mapping_smartmap;

  //read constrain file, construct mapping_stl
  std::string line;
  while (std::getline(lfs, line)) {
    line = ltp::strutils::chomp(line);
    if (line.size() == 0) {
      continue;
    }
    str_vec tokens = ltp::strutils::split(line);
    int num_tokens = tokens.size();
    std::string key = ltp::strutils::chartypes::sbc2dbc_x(tokens[0]);
    if( mapping_stl.find(key) == mapping_stl.end() ) {
      mapping_stl.insert( Map::value_type( key, str_vec() ) );
    }
    for (int i = 1; i < num_tokens; ++ i) {
      mapping_stl[key].push_back(tokens[i]);
    }
  }

  //sort & unique mapping_stl
  for(MapIt it = mapping_stl.begin() ;it != mapping_stl.end() ; ++it) {
    string key = it->first;
    str_vec & value = it->second;
    sort( value.begin(),value.end() );
    value.erase( unique(value.begin(),value.end()), value.end());
  }

  //traverse model external_lexicon, construct mapping_smartmap
  for (SmartMap<Bitset>::const_iterator itx = model->external_lexicon.begin();
    itx != model->external_lexicon.end();
    ++ itx ) {
      std::string key = itx.key();
      Bitset mask = (*(itx.value()));
      int_vec ones = mask.getbitones();
      str_vec value;
      for(int i=0;i<ones.size();i++) {
        std::string label = model->labels.at(ones[i]);
        value.push_back(label);
      }
      sort(value.begin(),value.end());
      mapping_smartmap.insert(Map::value_type(key,value) );
  }
  
  //mapping_stl size should equal to mapping_smartmap size 
  EXPECT_EQ(mapping_stl.size(), mapping_smartmap.size() );

  for(MapConstIt it = mapping_stl.begin() ;it != mapping_stl.end() ; ++it) {
    string key = it->first;
    str_vec  value_stl = it->second;
    MapConstIt key_it = mapping_smartmap.find(key);
    bool find = (mapping_smartmap.end() != key_it);
    EXPECT_TRUE(find);
    if(!find) {
      return;
    }
    str_vec  value_smartmap = key_it->second;
    bool value_size_equal = (value_stl.size() == value_smartmap.size() );
    EXPECT_TRUE(value_size_equal);
    if(!value_size_equal) {
      return;
    }
    for(int i=0;i<value_stl.size();i++) {
      EXPECT_EQ(value_stl[i],value_smartmap[i]);
    }
  }
  
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

