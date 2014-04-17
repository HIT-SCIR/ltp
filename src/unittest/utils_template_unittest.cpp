// Unittest case for functions in ltp::strutils module.
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "utils/template.hpp"

using namespace std;
using namespace ltp::utility;

TEST(template_unittest, test_template_basic) {
  Template   templates("1={test}");
  Template::Data data;
  data.set("test","basic");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1=basic",feat.c_str());
}

TEST(template_unittest, test_template_space) {
  Template   templates("1={test}");
  Template::Data data;
  data.set("test"," ");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1= ",feat.c_str());
}


TEST(template_unittest, test_template_empty) {
  Template   templates("1={test}");
  Template::Data data;
  data.set("test","");
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1=",feat.c_str());
}

TEST(template_unittest, test_template_null) {
  Template   templates("1={test}");
  Template::Data data;
  char * ptr = NULL;
  data.set("test",ptr);
  string feat;
  templates.render(data,feat);
  EXPECT_STREQ("1=",feat.c_str());
}


int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
