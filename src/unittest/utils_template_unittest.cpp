// Unittest case for functions in ltp::strutils module.
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "utils/template.hpp"

using namespace std;
using namespace ltp::utility;

TEST(template_unittest, test_template_basic) {
  Template T("1={test}");
  Template::Data data;
  data.set("test","basic");
  string feat;
  T.render(data,feat);
  EXPECT_STREQ("1=basic",feat.c_str());
}

TEST(template_unittest, test_template_space) {
  Template T("1={test}");
  Template::Data data;
  data.set("test"," ");
  string feat;
  T.render(data,feat);
  EXPECT_STREQ("1= ",feat.c_str());
}

TEST(template_unittest, test_template_empty) {
  Template T("1={test}");
  Template::Data data;
  data.set("test","");
  string feat;
  T.render(data,feat);
  EXPECT_STREQ("1=",feat.c_str());
}

TEST(template_unittest, test_template_concate) {
  /*Template T("1={slot1}{slot2}");
  Template::Data data;
  data.set("slot1", "-");
  data.set("slot2", "+");
  string feat;
  T.render(data, feat);
  EXPECT_STREQ("1=-+", feat.c_str());*/
}

TEST(template_unittest, test_template_chinese) {
  Template T("中文{test}");
  Template::Data data;
  data.set("test", "测试");
  string feat;
  T.render(data, feat);
  EXPECT_STREQ("中文测试", feat.c_str());
}

TEST(template_unittest, test_template_duplicate) {
  Template T("template-with-two-{{token}}");
  Template::Data data;
  data.set("token", "brackets");
  string feat;
  T.render(data, feat);
  EXPECT_STREQ("template-with-two-{brackets}", feat.c_str());
}

TEST(template_unittest, test_template_efficiency) {
  std::vector<Template *> repos;
  repos.push_back(new Template("1={w0}"));
  repos.push_back(new Template("2={p0}"));
  repos.push_back(new Template("3={w0}-{p0}"));
  repos.push_back(new Template("4={w1}"));
  repos.push_back(new Template("5={p1}"));
  repos.push_back(new Template("6={w1}-{p1}"));

  string payload;
  payload.reserve(128);
  long start_time = clock();
  int kNumRepeats = 1024 * 1024;
  int kNumTemplates = repos.size();

  for (int t = 0; t < 1024 * 1024; ++ t) {
    Template::Data data;
    data.set("w0", "am");
    data.set("p0", "v");
    data.set("w1", "I");
    data.set("p1", "r");
    for (int i = 0; i < repos.size(); ++ i) {
      Template* T = repos[i];
      T->render(data, payload);
    }
  }

  long throughput_per_millisecond = ((kNumRepeats * kNumTemplates)
      / ((clock() -start_time) / 1000));
  std::cerr << "#throughput: " <<throughput_per_millisecond << std::endl;
}

TEST(template_DeathTest, test_template_null) {
  Template T("1={test}");
  Template::Data data;
  // Till now (2014-04-30) this usage will result in a core
  // dump in LTP
  char* ptr = NULL;
  data.set("test", ptr);
  string feat;
  ASSERT_DEATH(T.render(data, feat), "");
}

