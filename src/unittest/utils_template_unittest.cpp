// Unittest case for functions in ltp::strutils module.
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "utils/template.hpp"

using namespace std;
using namespace ltp::utility;

TEST(template_unittest, test_template_basic) {
  Template templates("1={test}");
  Template::Data data;
  data.set("test","basic");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1=basic",feat.c_str());
}

TEST(template_unittest, test_template_space) {
  Template templates("1={test}");
  Template::Data data;
  data.set("test"," ");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1= ",feat.c_str());
}


TEST(template_unittest, test_template_empty) {
  Template templates("1={test}");
  Template::Data data;
  data.set("test","");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1=",feat.c_str());
}

TEST(template_DeathTest, test_template_null) {
  Template templates("1={test}");
  Template::Data data;
  // Till now (2014-04-30) this usage will result in a core
  // dump in LTP
  char * ptr = NULL;
  data.set("test", ptr);
  string feat;
  ASSERT_DEATH(templates.render(data, feat), "");
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

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
