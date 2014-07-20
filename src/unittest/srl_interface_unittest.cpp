// Provide some higher-level testcases
// For some unittest on the internal data structure and function, 
// please write in the ./srl_internal_unittest.cpp in same
// folder.
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "srl/SRL_DLL.h"

TEST(srl_interface_unittest, test_load_model_success) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  SRL_ReleaseResource();
}

TEST(srl_interface_unittest, test_load_model_fail) {
  int tag = SRL_LoadResource("/a/path/that/never/exist");
  EXPECT_EQ(-1, tag);
}

const char * kNormalWords[] = {"我", "是", "猫", "。"};
const char * kNormalPostags[] = {"r", "v", "n", "wp"};
const char * kNormalNes[] = {"O", "O", "O", "O"};
const int kNormalHeads[] = {1,-1,1,1};
const char * kNormalDeprels[] = {"SBV", "HED", "VOB", "WP"};
const int    kNumNormalWords = 4;

TEST(srl_interface_unittest, test_normal) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string > > parses;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srls;
  for (int i = 0; i < kNumNormalWords; ++ i) {
    words.push_back(kNormalWords[i]);
    tags.push_back(kNormalPostags[i]);
    nes.push_back(kNormalNes[i]);
    parses.push_back(std::make_pair(kNormalHeads[i], kNormalDeprels[i]));
  }
  int nr_words = DoSRL(words, tags , nes, parses ,srls);
  // tagged words should be greater than 4
  EXPECT_GT(nr_words, 0);
  SRL_ReleaseResource();
}

TEST(srl_interface_unittest, test_empty_list) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string > > parses;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srls;
 
  int nr_words = DoSRL(words, tags , nes, parses ,srls);
  EXPECT_EQ(0, nr_words);
  SRL_ReleaseResource();
}

TEST(srl_interface_unittest, test_empty_word) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string > > parses;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srls;
 
  for (int i = 0; i < kNumNormalWords; ++ i) {
    if (i == 2) {
      words.push_back("");
    } else {
      words.push_back(kNormalWords[i]);
    }
    tags.push_back(kNormalPostags[i]);
    nes.push_back(kNormalNes[i]);
    parses.push_back(std::make_pair(kNormalHeads[i], kNormalDeprels[i]));
  }

  int nr_words = DoSRL( words, tags, nes, parses, srls);
  EXPECT_EQ(-1, nr_words);
  SRL_ReleaseResource();
}

TEST(srl_interface_unittest, test_different_size) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string > > parses;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srls;
 
  for (int i = 0; i < kNumNormalWords; ++ i) {
    if (i != 2) {
      words.push_back(kNormalWords[i]);
    }
    tags.push_back(kNormalPostags[i]);
    tags.push_back(kNormalNes[i]);
    parses.push_back(std::make_pair(kNormalHeads[i], kNormalDeprels[i]));
  }

  int nr_words = DoSRL(words, tags, nes, parses, srls);
  EXPECT_EQ(-1, nr_words);
  SRL_ReleaseResource();
}


TEST(srl_interface_unittest, test_illeagel_head) {
  int tag = SRL_LoadResource("./ltp_data/srl");
  EXPECT_TRUE(0 == tag);
  std::vector<std::string> words;
  std::vector<std::string> tags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string > > parses;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srls;
 
  for (int i = 0; i < kNumNormalWords; ++ i) {
    tags.push_back(kNormalPostags[i]);
    words.push_back(kNormalWords[i]);
    nes.push_back(kNormalNes[i]);
    parses.push_back(std::make_pair(kNormalHeads[i], kNormalDeprels[i]));
  }
    parses[0].first = -2;

  int nr_words = DoSRL(words, tags, nes, parses, srls);
  EXPECT_EQ(-1, nr_words);
  SRL_ReleaseResource();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

