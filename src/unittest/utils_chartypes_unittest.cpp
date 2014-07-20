#include <iostream>
#include <gtest/gtest.h>
#include "utils/chartypes.hpp"

TEST(chartypes_unittest, miss_test) {
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_OTHER,
      ltp::strutils::chartypes::chartype("word"));
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_OTHER,
      ltp::strutils::chartypes::chartype("中文单词"));
}

TEST(chartypes_unittest, sbc_letter_test) {
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_LETTER,
      ltp::strutils::chartypes::chartype("a"));
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_LETTER,
      ltp::strutils::chartypes::chartype("A"));
}

TEST(chartypes_unittest, dbc_letter_test) {
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_LETTER,
      ltp::strutils::chartypes::chartype("ａ"));
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_LETTER,
      ltp::strutils::chartypes::chartype("Ａ"));
}

TEST(chartypes_unittest, sbc_digit_test) {
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_DIGIT,
      ltp::strutils::chartypes::chartype("1"));
}

TEST(chartypes_unittest, dbc_digit_test) {
  EXPECT_EQ(ltp::strutils::chartypes::CHAR_DIGIT,
      ltp::strutils::chartypes::chartype("１"));
}

const int kNumRepeat = 1000000;
const int kNumSpecialCharacters = 11;
const char * kSpecialCharacters[] = {
  "a", "Ａ", ",", "。", "，", "？", "1",
  "1234", "１", "letter", "中文单词",
};

// The chartype function should be able to parse at least 5,000 token
// per millisecond
TEST(chartypes_unittest, performance_test) {
  long start_time = clock();
  for (int i = 0; i < kNumRepeat; ++ i) {
    for (int j = 0; j < kNumSpecialCharacters; ++ j) {
      int x = ltp::strutils::chartypes::chartype(kSpecialCharacters[j]);
    }
  }
  long throughput_per_millisecond = ((kNumRepeat * kNumSpecialCharacters)
      / ((clock() -start_time) / 1000));
  EXPECT_LT(5000, throughput_per_millisecond);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
