// Unittest case for functions in ltp::strutils module.
#include <iostream>
#include <gtest/gtest.h>
#include "utils/strutils.hpp"

TEST(strutils_unittest, test_chomp_basic) {
  std::string testcase = ltp::strutils::chomp(" basic test \n");
  EXPECT_STREQ("basic test", testcase.c_str());
}

TEST(strutils_unittest, test_chomp_right) {
  std::string testcase = ltp::strutils::chomp("right strip\n\n");
  EXPECT_STREQ("right strip", testcase.c_str());
}

TEST(strutils_unittest, test_chomp_left) {
  std::string testcase = ltp::strutils::chomp("\n left strip");
  EXPECT_STREQ("left strip", testcase.c_str());
}

TEST(strutils_unittest, test_chomp_empty) {
  std::string testcase = ltp::strutils::chomp(" \n");
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
  EXPECT_STREQ("c", tokens[1].c_str());
}

// I am not quite sure about supporting this case, since each string
// is chomped before split in LTP.
TEST(strutils_unittest, test_split_space) {
  std::vector<std::string> tokens = ltp::strutils::split("  ");
  EXPECT_EQ(tokens.size(), 0);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
