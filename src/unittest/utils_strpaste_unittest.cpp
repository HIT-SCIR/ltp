#include <iostream>
#include <gtest/gtest.h>
#include "utils/strpaste.hpp"

TEST(strpaste_unittest, test_paste_1) {
  std::string out;
  ltp::strutils::paste(out, "1");
  EXPECT_STREQ("1", out.c_str());
}

TEST(strpaste_unittest, test_paste_2) {
  std::string out;
  ltp::strutils::paste(out, "1", "2");
  EXPECT_STREQ("12", out.c_str());
}

TEST(strpaste_unittest, test_paste_3) {
  std::string out;
  ltp::strutils::paste(out, "1", "2", "3");
  EXPECT_STREQ("123", out.c_str());
}

TEST(strpaste_unittest, test_paste_4) {
  std::string out;
  ltp::strutils::paste(out, "1", "2", "3", "4");
  EXPECT_STREQ("1234", out.c_str());
}

TEST(strpaste_unittest, test_paste_5) {
  std::string out;
  ltp::strutils::paste(out, "1", "2", "3", "4", "5");
  EXPECT_STREQ("12345", out.c_str());
}

