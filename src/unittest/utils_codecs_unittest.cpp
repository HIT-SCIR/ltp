#include <iostream>
#include <gtest/gtest.h>
#include "utils/codecs.hpp"

TEST(codecs_unittest, standard_test) {
  std::string test = "中文测试UTF-8编码测试";
  std::vector<std::string> chars;

  int ret = ltp::strutils::codecs::decode(test, chars);
  EXPECT_EQ(13, ret);
  EXPECT_EQ(13, chars.size());
  EXPECT_STREQ("中", chars[0].c_str());
  EXPECT_STREQ("U", chars[4].c_str());
  EXPECT_STREQ("-", chars[7].c_str());
}

TEST(codecs_unittest, const_iterator_test) {
  std::string test = "中文测试UTF-8编码测试";
  for (ltp::strutils::codecs::iterator i(test); !i.is_end(); ++ i) {
    std::string s = test.substr(i->first, i->second - i->first);
  }
}

TEST(codecs_unittest, initial_character_test) {
  std::string test;
  bool success = ltp::strutils::codecs::initial("中文测试UTF-8编码测试", test);
  EXPECT_EQ(true, success);
  EXPECT_STREQ("中", test.c_str());
}

TEST(codecs_unittest, tail_character_test) {
  std::string test;
  bool success = ltp::strutils::codecs::tail("中文测试UTF-8编码测试", test);
  EXPECT_EQ(true, success);
  EXPECT_STREQ("试", test.c_str());
}

TEST(codecs_unittest, length_test) {
  int length = ltp::strutils::codecs::length("中文测试UTF-8编码测试");
  EXPECT_EQ(13, length);
}

TEST(codecs_unittest, is_unicode_punctuation) {
  EXPECT_TRUE(ltp::strutils::codecs::is_unicode_punctuation("/"));
  EXPECT_TRUE(ltp::strutils::codecs::is_unicode_punctuation("?"));
  EXPECT_TRUE(ltp::strutils::codecs::is_unicode_punctuation("。"));
  EXPECT_TRUE(ltp::strutils::codecs::is_unicode_punctuation("，"));
  EXPECT_FALSE(ltp::strutils::codecs::is_unicode_punctuation("社"));
  EXPECT_FALSE(ltp::strutils::codecs::is_unicode_punctuation("3"));
}

