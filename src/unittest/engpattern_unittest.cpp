#include <iostream>
#include <gtest/gtest.h>
#include <string>
#include "boost/regex.hpp"
#include "segmentor/rulebase.h"

TEST(engpattern_unittest, english_word) {
  std::string word = "78G"; 
  EXPECT_EQ(true, boost::regex_match(word,ltp::segmentor::rulebase::engpattern));

  word = "78-d";
  EXPECT_EQ(true, boost::regex_match(word,ltp::segmentor::rulebase::engpattern));

  word = "md3243";
  EXPECT_EQ(true, boost::regex_match(word,ltp::segmentor::rulebase::engpattern));

  word = "md-3243";
  EXPECT_EQ(true, boost::regex_match(word,ltp::segmentor::rulebase::engpattern));
}

TEST(engpattern_unittest, number) {
  std::string word = "1997"; 
  EXPECT_EQ(false, boost::regex_match(word,ltp::segmentor::rulebase::engpattern));
}


int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
