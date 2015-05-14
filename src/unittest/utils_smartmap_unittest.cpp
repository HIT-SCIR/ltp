#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "utils/smartmap.hpp"

using ltp::utility::SmartMap;

TEST(smartmap_unittest, continuous_loading) {
  SmartMap<int> dict1; dict1.set("dict1", 1);
  SmartMap<int> dict2; dict2.set("dict2", 2);

  std::ofstream ofs("smartmap_unittest.dummyfile", std::ofstream::binary);
  dict1.dump(ofs);
  dict2.dump(ofs);
  ofs.close();

  std::ifstream ifs("smartmap_unittest.dummyfile", std::ifstream::binary);
  SmartMap<int> dict3; dict3.load(ifs);
  SmartMap<int> dict4; dict4.load(ifs);

  int val;
  dict3.get("dict1", val); EXPECT_EQ(val, 1);
  dict4.get("dict2", val); EXPECT_EQ(val, 2);
}
