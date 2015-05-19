#include <iostream>
#include <gtest/gtest.h>
#include "utils/sbcdbc.hpp"

TEST(sbcdbc_unittest, miss_test) {
  std::string tmp;
  ltp::strutils::chartypes::sbc2dbc("w", tmp);
  EXPECT_STREQ("ｗ", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc("1", tmp);
  EXPECT_STREQ("１", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc(" ", tmp);
  EXPECT_STREQ("　", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc("!", tmp);
  EXPECT_STREQ("！", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc(",", tmp);
  EXPECT_STREQ("，", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc("?", tmp);
  EXPECT_STREQ("？", tmp.c_str());
  ltp::strutils::chartypes::sbc2dbc("#", tmp);
  EXPECT_STREQ("＃", tmp.c_str());
}

// The chartype function should be able to parse at least 5,000 token
// per millisecond
TEST(sbcdbc_unittest, performance_sbc2dbc1_test) {
  const int kNumRepeat = 10000000;
  long start_time = clock();
  for (int i = 0; i < kNumRepeat; ++ i) {
    ltp::strutils::chartypes::sbc2dbc("w");
  }
  long throughput_per_millisecond = (kNumRepeat / ((clock() -start_time) / 1000));
  std::cout << "#throughput ltp::strutils::chartypes::sbc2dbc : "
    << throughput_per_millisecond << std::endl;
  EXPECT_LT(5000, throughput_per_millisecond);
}

TEST(sbcdbc_unittest, performance_sbc2dbc2_test) {
  const int kNumRepeat = 10000000;
  long start_time = clock();
  std::string tmp; tmp.reserve(10);
  for (int i = 0; i < kNumRepeat; ++ i) {
    ltp::strutils::chartypes::sbc2dbc("w", tmp);
  }
  long throughput_per_millisecond = (kNumRepeat / ((clock() -start_time) / 1000));
  std::cout << "#throughput ltp::strutils::chartypes::sbc2dbc : "
    << throughput_per_millisecond << std::endl;
  EXPECT_LT(5000, throughput_per_millisecond);
}

