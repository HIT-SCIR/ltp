// Provide some higher-level testcases
// For some unittest on the internal data structure and function, 
// please write in the ./parser_internal_unittest.cpp in same
// folder.
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "parser/parser_dll.h"
#include "utils/strutils.hpp"

TEST(parser_interface_unittest, test_load_model_success) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);
  parser_release_parser(engine);
}

TEST(parser_interface_unittest, test_load_model_fail) {
  void * engine = parser_create_parser("/a/path/that/never/exist");
  EXPECT_EQ(NULL, engine);
}

const char * kNormalWords[] = {"我", "是", "猫", "。"};
const char * kNormalPostags[] = {"r", "v", "n", "wp"};
const int    kNumNormalWords = 4;

TEST(parser_interface_unittest, test_normal) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
  for (int i = 0; i < kNumNormalWords; ++ i) {
    words.push_back(kNormalWords[i]);
    tags.push_back(kNormalPostags[i]);
  }
  int nr_words = parser_parse(engine, words, tags ,heads ,deprels);
  // tagged words should be greater than 4
  EXPECT_GT(nr_words, 0);
  parser_release_parser(engine);
}

TEST(parser_interface_unittest, test_empty_list) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
 
  int nr_words = parser_parse(engine, words, tags ,heads ,deprels);
  EXPECT_EQ(0, nr_words);
  parser_release_parser(engine);
}

TEST(parser_interface_unittest, test_empty_word) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
 
  for (int i = 0; i < kNumNormalWords; ++ i) {
    if (i == 2) {
      words.push_back("");
    } else {
      words.push_back(kNormalWords[i]);
    }
    tags.push_back(kNormalPostags[i]);
  }

  int nr_words = parser_parse(engine, words, tags, heads, deprels);
  EXPECT_EQ(0, nr_words);
  parser_release_parser(engine);
}

TEST(parser_interface_unittest, test_empty_tag) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
 
  for (int i = 0; i < kNumNormalWords; ++ i) {
    if (i == 2) {
      tags.push_back("");
    } else {
      tags.push_back(kNormalPostags[i]);
    }
    words.push_back(kNormalWords[i]);
  }

  int nr_words = parser_parse(engine, words, tags, heads, deprels);
  EXPECT_EQ(0, nr_words);
  parser_release_parser(engine);
}


TEST(parser_interface_unittest, test_speed) {
  void * engine = parser_create_parser("./ltp_data/parser.model");
  EXPECT_TRUE(NULL != engine);

  std::ifstream ifs("./test_data/unittest/test_data.postaggered");
  std::string line;
  std::string word;
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
 
  int nr_tokens = 0;

  long start_time = clock();
  while (std::getline(ifs, line, '\n')) {
    std::stringstream S(line);
    words.clear();
    tags.clear();
    heads.clear();
    deprels.clear();
    while (S >> word) {
      std::vector<std::string> sep = ltp::strutils::rsplit_by_sep(word, "_", 1);
      if(sep.size()==2) {
        words.push_back(sep[0]);
        tags.push_back(sep[1]);
      } else {
        std::cerr << word<<std::endl;
        return;
      }
    }
    parser_parse(engine, words, tags, heads, deprels);
    nr_tokens += words.size();
  }
  double throughput_per_millisecond = (nr_tokens / ((clock() -start_time) / 1000.));
  std::cerr << throughput_per_millisecond << std::endl;
  parser_release_parser(engine);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

