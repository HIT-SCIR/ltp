#include <iostream>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include "segmentor/preprocessor.h"
#include "segmentor/decoder.h"
#include "segmentor/partial_segmentation.h"
#include "segmentor/settings.h"
#include "segmentor/segmentor.h"
#include "utils/chartypes.hpp"
#include "boost/regex.hpp"

using ltp::segmentor::Preprocessor;
using ltp::segmentor::PartialSegmentationUtils;
using ltp::segmentor::SegmentationConstrain;
using ltp::segmentor::Segmentor;
using ltp::segmentor::__b_id__;
using ltp::segmentor::__i_id__;
using ltp::segmentor::__e_id__;
using ltp::segmentor::__s_id__;
using ltp::strutils::chartypes::CHAR_OTHER;
using ltp::strutils::chartypes::CHAR_PUNC;

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

TEST(segmentor_unittest, preprocess6) {
  Preprocessor preprocessor;

  std::vector<std::string> raw_forms;
  std::vector<std::string> forms;
  std::vector<int> chartypes;
  int retval = preprocessor.preprocess("套头衫/Jacquad Sweater",
      raw_forms, forms, chartypes);
  EXPECT_EQ(retval, 6);

  EXPECT_EQ(raw_forms.size(), 6);
  EXPECT_STREQ(raw_forms[3].c_str(), "/");
  EXPECT_STREQ(raw_forms[4].c_str(), "Jacquad");
  EXPECT_STREQ(raw_forms[5].c_str(), "Sweater");

  EXPECT_EQ(chartypes.size(), 6);
  EXPECT_EQ(chartypes[3], CHAR_PUNC|preprocessor.HAS_ENG_ON_RIGHT);
  EXPECT_EQ(chartypes[4], Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_RIGHT);
  EXPECT_EQ(chartypes[5], Preprocessor::CHAR_ENG|preprocessor.HAS_SPACE_ON_LEFT);
}

TEST(segmentor_unittest, partial_segment_split_by_tag_whole_sentence) {
  std::vector<std::string> output;
  PartialSegmentationUtils::split_by_partial_tag(
      "这是测试句子", output);

  EXPECT_EQ(output.size(), 1);
}

TEST(segmentor_unittest, partial_segment_split_by_tag_words) {
  std::vector<std::string> output;
  PartialSegmentationUtils::split_by_partial_tag(
      "这 是 测试 句子", output);

  EXPECT_EQ(output.size(), 4);
  EXPECT_STREQ(output[0].c_str(), "这");
  EXPECT_STREQ(output[1].c_str(), "是");
  EXPECT_STREQ(output[2].c_str(), "测试");
  EXPECT_STREQ(output[3].c_str(), "句子");
}


TEST(segmentor_unittest, partial_segment_split_by_tag_canonical) {
  std::vector<std::string> output;
  PartialSegmentationUtils::split_by_partial_tag(
      "这是<ltp:partial>测试</ltp:partial>句子", output);

  EXPECT_EQ(output.size(), 3);
  EXPECT_STREQ(output[0].c_str(), "这是");
  EXPECT_STREQ(output[1].c_str(), "<ltp:partial>测试</ltp:partial>");
  EXPECT_STREQ(output[2].c_str(), "句子");
}

TEST(segmentor_unittest, partial_segment_split_by_tag_combo) {
  std::vector<std::string> output;
  PartialSegmentationUtils::split_by_partial_tag(
      "这是<ltp:partial>测试</ltp:partial>句<ltp:word>子</ltp:word>",
      output);

  EXPECT_EQ(output.size(), 4);
  EXPECT_STREQ(output[0].c_str(), "这是");
  EXPECT_STREQ(output[1].c_str(), "<ltp:partial>测试</ltp:partial>");
  EXPECT_STREQ(output[2].c_str(), "句");
  EXPECT_STREQ(output[3].c_str(), "<ltp:word>子</ltp:word>");
}

TEST(segmentor_unittest, partial_segment_split_by_tag_combo2) {
  std::vector<std::string> output;
  PartialSegmentationUtils::split_by_partial_tag(
      "这是<ltp:partial>测试</ltp:partial><ltp:word>句子</ltp:word>",
      output);

  EXPECT_EQ(output.size(), 3);
  EXPECT_STREQ(output[0].c_str(), "这是");
  EXPECT_STREQ(output[1].c_str(), "<ltp:partial>测试</ltp:partial>");
  EXPECT_STREQ(output[2].c_str(), "<ltp:word>句子</ltp:word>");
}

TEST(segmentor_unittest, partial_segment_split_by_tag_nested) {
  std::vector<std::string> output;
  EXPECT_EQ(-1, PartialSegmentationUtils::split_by_partial_tag(
      "这是<ltp:partial>测试<ltp:word>句</ltp:word>子</ltp:partial>",
      output));
}

TEST(segmentor_unittest, partial_segment_split_by_tag_duplicated) {
  std::vector<std::string> output;
  EXPECT_EQ(-1, PartialSegmentationUtils::split_by_partial_tag(
      "这是<ltp:partial>测试<ltp:partial>句子</ltp:partial>",
      output));
}

TEST(segmentor_unittest, partial_segment_trim_tag) {
  std::string word;
  PartialSegmentationUtils::trim_partial_tag(
      "<ltp:partial>句子</ltp:partial>", word);
  EXPECT_STREQ("句子", word.c_str());
}

TEST(segmentor_unittest, segmentor_build_words) {
  // canonical build words.
  std::vector<std::string> chars;
  chars.push_back( "这" );
  chars.push_back( "是" );
  chars.push_back( "测" );
  chars.push_back( "试" );
  chars.push_back( "句" );
  chars.push_back( "子" );

  std::vector<int> tags;
  tags.push_back( __s_id__ );
  tags.push_back( __s_id__ );
  tags.push_back( __b_id__ );
  tags.push_back( __e_id__ );
  tags.push_back( __b_id__ );
  tags.push_back( __e_id__ );

  Segmentor app;
  std::vector<std::string> result;
  app.build_words(chars, tags, result);

  EXPECT_EQ(4, result.size());
  EXPECT_STREQ( "这", result[0].c_str() );
  EXPECT_STREQ( "是", result[1].c_str() );
  EXPECT_STREQ( "测试", result[2].c_str() );
  EXPECT_STREQ( "句子", result[3].c_str() );
}

TEST(segmentor_unittest, segmentor_build_words2) {
  std::vector<std::string> chars;
  chars.push_back( "这" );
  chars.push_back( "是" );
  chars.push_back( "测" );
  chars.push_back( "试" );
  chars.push_back( "句" );
  chars.push_back( "子" );

  std::vector<int> tags;
  tags.push_back( __s_id__ );
  tags.push_back( __e_id__ );
  tags.push_back( __e_id__ );
  tags.push_back( __b_id__ );
  tags.push_back( __i_id__ );
  tags.push_back( __i_id__ );

  Segmentor app;
  std::vector<std::string> result;
  app.build_words(chars, tags, result);

  EXPECT_EQ(2, result.size());
  EXPECT_STREQ( "这是测", result[0].c_str() );
  EXPECT_STREQ( "试句子", result[1].c_str() );
}

TEST(segmentor_unittest, constrain1) {
  SegmentationConstrain con;
  EXPECT_TRUE(con.can_tran(0, 2));
}

