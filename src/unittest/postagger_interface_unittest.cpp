// Provide some higher-level testcases
// For some unittest on the internal data structure and function, 
// please write in the ./postagger_internal_unittest.cpp in same
// folder.
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "postagger/postag_dll.h"

TEST(postag_interface_unittest, test_load_model_success) {
  void * engine = postagger_create_postagger("./ltp_data/pos.model");
  EXPECT_TRUE(NULL != engine);
  postagger_release_postagger(engine);
}

TEST(postag_interface_unittest, test_load_model_fail) {
  void * engine = postagger_create_postagger("/a/path/that/never/exist");
  EXPECT_EQ(NULL, engine);
}

const char * kNormalWords[] = {"我", "是", "猫", "。"};
const int    kNumNormalWords = 4;

TEST(postag_interface_unittest, test_normal) {
  void * engine = postagger_create_postagger("./ltp_data/pos.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  for (int i = 0; i < kNumNormalWords; ++ i) {
    words.push_back(kNormalWords[i]);
  }
  int nr_words = postagger_postag(engine, words, tags);
  // tagged words should be greater than 4
  EXPECT_GT(nr_words, 0);
  postagger_release_postagger(engine);
}

TEST(postag_interface_unittest, test_empty_list) {
  void * engine = postagger_create_postagger("./ltp_data/pos.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  int nr_words = postagger_postag(engine, words, tags);
  EXPECT_EQ(0, nr_words);
  postagger_release_postagger(engine);
}

TEST(postag_interface_unittest, test_empty_word) {
  void * engine = postagger_create_postagger("./ltp_data/pos.model");
  EXPECT_TRUE(NULL != engine);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  for (int i = 0; i < kNumNormalWords; ++ i) {
    if (i == 2) {
      words.push_back("");
    } else {
      words.push_back(kNormalWords[i]);
    }
  }
  int nr_words = postagger_postag(engine, words, tags);
  EXPECT_EQ(0, nr_words);
  postagger_release_postagger(engine);
}

TEST(postag_interface_unittest, test_speed) {
  void * engine = postagger_create_postagger("./ltp_data/pos.model");
  EXPECT_TRUE(NULL != engine);

  std::ifstream ifs("./test_data/unittest/test_data.segmented");
  std::string line;
  std::string word;
  std::vector<std::string> words;
  std::vector<std::string> tags;
  int nr_tokens = 0;

  long start_time = clock();
  while (std::getline(ifs, line, '\n')) {
    std::stringstream S(line);
    words.clear();
    tags.clear();
    while (S >> word) {
      words.push_back(word);
    }
    postagger_postag(engine, words, tags);
    nr_tokens += words.size();
  }
  double throughput_per_millisecond = (nr_tokens / ((clock() -start_time) / 1000.));
  std::cerr << throughput_per_millisecond << std::endl;
  postagger_release_postagger(engine);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

