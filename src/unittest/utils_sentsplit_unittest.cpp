#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "utils/sentsplit.hpp"

TEST(sentsplit_unittest, test1) {
  std::vector<std::string> sentences;

  ltp::Chinese::split_sentence("测试句子一。", sentences);
  EXPECT_EQ(sentences.size(), 1);
  EXPECT_STREQ(sentences[0].c_str(), "测试句子一。");
}

TEST(sentsplit_unittest, test2) {
  std::vector<std::string> sentences;

  ltp::Chinese::split_sentence("测试句子一。测试句子二。", sentences);
  EXPECT_EQ(sentences.size(), 2);
  EXPECT_STREQ(sentences[0].c_str(), "测试句子一。");
  EXPECT_STREQ(sentences[1].c_str(), "测试句子二。");
}

TEST(sentsplit_unittest, test3) {
  std::vector<std::string> sentences;

  ltp::Chinese::split_sentence("测试句子一。测试句子二。”", sentences);
  EXPECT_EQ(sentences.size(), 2);
  EXPECT_STREQ(sentences[0].c_str(), "测试句子一。");
  EXPECT_STREQ(sentences[1].c_str(), "测试句子二。”");
}

TEST(sentsplit_unittest, test4) {
  std::vector<std::string> sentences;

  ltp::Chinese::split_sentence("测试句子一。”测试句子二。", sentences);
  EXPECT_EQ(sentences.size(), 2);
  EXPECT_STREQ(sentences[0].c_str(), "测试句子一。”");
  EXPECT_STREQ(sentences[1].c_str(), "测试句子二。");
}

TEST(sentsplit_unittest, test5) {
  std::vector<std::string> sentences;

  ltp::Chinese::split_sentence("3部长号 （2部高音，1部低音）", sentences);
  EXPECT_EQ(sentences.size(), 1);
  EXPECT_STREQ(sentences[0].c_str(), "3部长号 （2部高音，1部低音）");
}
