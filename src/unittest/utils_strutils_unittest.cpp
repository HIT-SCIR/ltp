// Unittest case for functions in ltp::strutils module.
#include <iostream>
#include <gtest/gtest.h>
#include "utils/strutils.hpp"

TEST(strutils_unittest, test_trim_basic) {
  std::string testcase = " basic test \n";
  ltp::strutils::trim(testcase);
  EXPECT_STREQ("basic test", testcase.c_str());
}

TEST(strutils_unittest, test_trim_right) {
  std::string testcase = "right strip\n\n";
  ltp::strutils::trim(testcase);
  EXPECT_STREQ("right strip", testcase.c_str());
}

TEST(strutils_unittest, test_trim_left) {
  std::string testcase ="\n left strip";
  ltp::strutils::trim(testcase);
  EXPECT_STREQ("left strip", testcase.c_str());
}

TEST(strutils_unittest, test_trim_empty) {
  std::string testcase = " \n";
  ltp::strutils::trim(testcase);
  EXPECT_STREQ("", testcase.c_str());
}

TEST(strutils_unittest, test_trim_copy_basic) {
  std::string testcase = ltp::strutils::trim_copy(" basic test \n");
  EXPECT_STREQ("basic test", testcase.c_str());
}

TEST(strutils_unittest, test_trim_copy_right) {
  std::string testcase = ltp::strutils::trim_copy("right strip\n\n");
  EXPECT_STREQ("right strip", testcase.c_str());
}

TEST(strutils_unittest, test_trim_copy_left) {
  std::string testcase = ltp::strutils::trim_copy("\n left strip");
  EXPECT_STREQ("left strip", testcase.c_str());
}

TEST(strutils_unittest, test_trim_copy_empty) {
  std::string testcase = ltp::strutils::trim_copy(" \n");
  EXPECT_STREQ("", testcase.c_str());
}

TEST(strutils_unittest, test_split_basic) {
  std::vector<std::string> tokens = ltp::strutils::split("a b c");
  EXPECT_EQ(tokens.size(), 3);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b", tokens[1].c_str());
  EXPECT_STREQ("c", tokens[2].c_str());
}

TEST(strutils_unittest, test_split_contiguous) {
  std::vector<std::string> tokens = ltp::strutils::split("a  b   c");
  EXPECT_EQ(tokens.size(), 3);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b", tokens[1].c_str());
  EXPECT_STREQ("c", tokens[2].c_str());
}

TEST(strutils_unittest, test_split_empty) {
  std::vector<std::string> tokens = ltp::strutils::split("");
  EXPECT_EQ(tokens.size(), 0);
}

// I am not quite sure about supporting this case, since each string
// is chomped before split in LTP.
TEST(strutils_unittest, test_split_leading_ending_space) {
  std::vector<std::string> tokens = ltp::strutils::split(" a b c ");
  EXPECT_EQ(tokens.size(), 3);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b", tokens[1].c_str());
  EXPECT_STREQ("c", tokens[2].c_str());
}

// I am not quite sure about supporting this case, since each string
// is chomped before split in LTP.
TEST(strutils_unittest, test_split_space) {
  std::vector<std::string> tokens = ltp::strutils::split("  ");
  EXPECT_EQ(tokens.size(), 0);
}

TEST(strutils_unittest, test_split_limit_num) {
  std::vector<std::string> tokens = ltp::strutils::split("a b c", 1);
  EXPECT_EQ(tokens.size(), 2);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b c", tokens[1].c_str());
}

TEST(strutils_unittest, test_split_by_sep) {
  std::vector<std::string> tokens = ltp::strutils::split_by_sep("a-b-c", "-");
  EXPECT_EQ(tokens.size(), 3);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b", tokens[1].c_str());
  EXPECT_STREQ("c", tokens[2].c_str());
}

TEST(strutils_unittest, test_split_by_sep_nonexist) {
  std::vector<std::string> tokens = ltp::strutils::split_by_sep("a-b-c", "|");
  EXPECT_EQ(tokens.size(), 1);
  EXPECT_STREQ("a-b-c", tokens[0].c_str());
}

TEST(strutils_unittest, test_split_by_sep_and_limit_num) {
  std::vector<std::string> tokens = ltp::strutils::split_by_sep("a-b-c", "-", 1);
  EXPECT_EQ(tokens.size(), 2);
  EXPECT_STREQ("a", tokens[0].c_str());
  EXPECT_STREQ("b-c", tokens[1].c_str());
}

TEST(strutils_unittest, test_rsplit_limit_num) {
  std::vector<std::string> tokens = ltp::strutils::rsplit("a b c", 1);
  EXPECT_EQ(tokens.size(), 2);
  EXPECT_STREQ("a b", tokens[0].c_str());
  EXPECT_STREQ("c", tokens[1].c_str());
}

