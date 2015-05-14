#define private public
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "segmentor/preprocessor.h"
#include "segmentor/decoder.h"
#include "utils/chartypes.hpp"
#include "boost/regex.hpp"

using ltp::segmentor::Preprocessor;
using ltp::segmentor::SegmentorConstrain;
using ltp::strutils::chartypes::CHAR_OTHER;

TEST(engpattern_unittest, english_word) {
  Preprocessor preprocessor;

  std::string word = "78G";
  EXPECT_TRUE(boost::regex_match(word, preprocessor.eng_regex));

  word = "78-d";
  EXPECT_TRUE(boost::regex_match(word, preprocessor.eng_regex));

  word = "md3243";
  EXPECT_TRUE(boost::regex_match(word, preprocessor.eng_regex));

  word = "md-3243";
  EXPECT_TRUE(boost::regex_match(word, preprocessor.eng_regex));
}

TEST(engpattern_unittest, number) {
  Preprocessor preprocessor;

  std::string word = "1997";
  EXPECT_FALSE(boost::regex_match(word, preprocessor.eng_regex));
}

TEST(segmentor_unittest, preprocess1) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("中文单词", raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 4);

  EXPECT_EQ(raw_forms.size(), 4);
  EXPECT_STREQ(raw_forms[0].c_str(), "中");
  EXPECT_STREQ(raw_forms[1].c_str(), "文");
  EXPECT_STREQ(raw_forms[2].c_str(), "单");
  EXPECT_STREQ(raw_forms[3].c_str(), "词");

  EXPECT_EQ(forms.size(), 4);
  EXPECT_STREQ(forms[0].c_str(), "中");
  EXPECT_STREQ(forms[1].c_str(), "文");
  EXPECT_STREQ(forms[2].c_str(), "单");
  EXPECT_STREQ(forms[3].c_str(), "词");

  EXPECT_EQ(chartypes.size(), 4);
  EXPECT_EQ(chartypes[0], CHAR_OTHER);
  EXPECT_EQ(chartypes[1], CHAR_OTHER);
  EXPECT_EQ(chartypes[2], CHAR_OTHER);
  EXPECT_EQ(chartypes[3], CHAR_OTHER);
}

TEST(segmentor_unittest, preprocess2) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("中文 单  词", raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 4);

  EXPECT_EQ(raw_forms.size(), 4);
  EXPECT_STREQ(raw_forms[0].c_str(), "中");
  EXPECT_STREQ(raw_forms[1].c_str(), "文");
  EXPECT_STREQ(raw_forms[2].c_str(), "单");
  EXPECT_STREQ(raw_forms[3].c_str(), "词");

  EXPECT_EQ(forms.size(), 4);
  EXPECT_STREQ(forms[0].c_str(), "中");
  EXPECT_STREQ(forms[1].c_str(), "文");
  EXPECT_STREQ(forms[2].c_str(), "单");
  EXPECT_STREQ(forms[3].c_str(), "词");

  EXPECT_EQ(chartypes.size(), 4);
  EXPECT_EQ(chartypes[0], CHAR_OTHER);
  EXPECT_EQ(chartypes[1], CHAR_OTHER|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[2], 
      CHAR_OTHER|preprocessor.HAS_SPACE_ON_LEFT|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[3], CHAR_OTHER|preprocessor.HAS_SPACE_ON_LEFT);
}

TEST(segmentor_unittest, preprocess3) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("中文mix单词", raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 5);

  EXPECT_EQ(raw_forms.size(), 5);
  EXPECT_STREQ(raw_forms[0].c_str(), "中");
  EXPECT_STREQ(raw_forms[1].c_str(), "文");
  EXPECT_STREQ(raw_forms[2].c_str(), "mix");
  EXPECT_STREQ(raw_forms[3].c_str(), "单");
  EXPECT_STREQ(raw_forms[4].c_str(), "词");

  EXPECT_EQ(forms.size(), 5);
  EXPECT_STREQ(forms[0].c_str(), "中");
  EXPECT_STREQ(forms[1].c_str(), "文");
  EXPECT_STREQ(forms[2].c_str(), "_eng_");
  EXPECT_STREQ(forms[3].c_str(), "单");
  EXPECT_STREQ(forms[4].c_str(), "词");

  EXPECT_EQ(chartypes.size(), 5);
  EXPECT_EQ(chartypes[0], CHAR_OTHER);
  EXPECT_EQ(chartypes[1], CHAR_OTHER|preprocessor.HAS_ENG_ON_RIGHT);
  EXPECT_EQ(chartypes[2], Preprocessor::CHAR_ENG);
  EXPECT_EQ(chartypes[3], CHAR_OTHER|preprocessor.HAS_ENG_ON_LEFT);
  EXPECT_EQ(chartypes[4], CHAR_OTHER);
}

TEST(segmentor_unittest, preprocess4) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("i like reading", raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 3);

  EXPECT_EQ(raw_forms.size(), 3);
  EXPECT_STREQ(raw_forms[0].c_str(), "i");
  EXPECT_STREQ(raw_forms[1].c_str(), "like");
  EXPECT_STREQ(raw_forms[2].c_str(), "reading");

  EXPECT_EQ(forms.size(), 3);
  EXPECT_STREQ(forms[0].c_str(), "_eng_");
  EXPECT_STREQ(forms[1].c_str(), "_eng_");
  EXPECT_STREQ(forms[2].c_str(), "_eng_");

  EXPECT_EQ(chartypes.size(), 3);
  EXPECT_EQ(chartypes[0], Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[1],
      Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_LEFT|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[2], Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_LEFT);
}

TEST(segmentor_unittest, preprocess5) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("Python日报 2015-05-12 http://t.cn/RAsCOoK",
      raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 5);

  EXPECT_EQ(raw_forms.size(), 5);
  EXPECT_STREQ(raw_forms[0].c_str(), "Python");
  EXPECT_STREQ(raw_forms[3].c_str(), "2015-05-12");
  EXPECT_STREQ(raw_forms[4].c_str(), "http://t.cn/RAsCOoK");

  EXPECT_EQ(forms.size(), 5);
  EXPECT_STREQ(forms[0].c_str(), "_eng_");
  EXPECT_STREQ(forms[3].c_str(), "_eng_");
  EXPECT_STREQ(forms[4].c_str(), "_uri_");

  EXPECT_EQ(chartypes.size(), 5);
  EXPECT_EQ(chartypes[0], Preprocessor::CHAR_ENG);
  EXPECT_EQ(chartypes[2], CHAR_OTHER|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[3],
      Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_LEFT|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[4], Preprocessor::CHAR_URI|preprocessor.HAS_SPACE_ON_LEFT);
}

TEST(segmentor_unittest, constrain1) {
  SegmentorConstrain con;
  EXPECT_TRUE(con.can_tran(0, 2));
}

