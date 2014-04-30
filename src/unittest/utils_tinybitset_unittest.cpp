#include <iostream>
#include <gtest/gtest.h>
#include "utils/tinybitset.hpp"

using namespace std;
using namespace ltp::utility;

const int kNormalCaseNum = 8;

const int kNormalCase[8] = {0,1,3,32,34,57,89,127};

TEST(tinybitset_unittest, test_bitset_init) {
  Bitset bitset;
  bool res = bitset.empty();
  EXPECT_EQ(true, res);
  vector<int> bitones = bitset.getbitones();
  EXPECT_EQ(0,bitones.size());
}

TEST(tinybitset_unittest, test_bitset_allsetones) {
  Bitset bitset;
  bitset.allsetones();
  bool res = bitset.empty();
  EXPECT_EQ(true, res);
  vector<int> bitones = bitset.getbitones();
  EXPECT_EQ(128,bitones.size());
  for(int i=0;i<bitones.size();i++){
    EXPECT_EQ(i,bitones[i]);
  }
}

TEST(tinybitset_unittest, test_bitset_boundrytest) {
  Bitset bitset;
  bool res = bitset.set(-1);
  EXPECT_EQ(0,res);
  res = bitset.set(128);
  EXPECT_EQ(0,res);
  res = bitset.set(0);
  EXPECT_EQ(1,res);
  res = bitset.set(127);
  EXPECT_EQ(1,res);

  res = bitset.get(-1);
  EXPECT_EQ(0,res);
  res = bitset.get(128);
  EXPECT_EQ(0,res);
  res = bitset.get(0);
  EXPECT_EQ(1,res);
  res = bitset.get(127);
  EXPECT_EQ(1,res);
}

TEST(tinybitset_unittest, test_bitset_normaltest) {
  Bitset bitset;
  bool res;
  for(int i=0;i<kNormalCaseNum;i++){
    bitset.set(kNormalCase[i]);
    res = bitset.get(kNormalCase[i]);
    EXPECT_EQ(1,res);
    res = bitset.empty();
    EXPECT_EQ(0,res);
  }
  vector<int> bitones = bitset.getbitones();
  EXPECT_EQ(kNormalCaseNum,bitones.size());
  for(int i=0;i<kNormalCaseNum;i++){
    EXPECT_EQ(kNormalCase[i],bitones[i]);
  }
}

TEST(tinybitset_unittest, test_bitset_mergetest) {
  Bitset bitset1;
  Bitset bitset2;
  bitset1.merge(bitset2);
  bool res = bitset1.empty();
  EXPECT_EQ(1,res);
  res =  bitset2.empty();
  EXPECT_EQ(1,res);
  vector<int> bitones = bitset1.getbitones();
  EXPECT_EQ(0,bitones.size());
  bitones = bitset2.getbitones();
  EXPECT_EQ(0,bitones.size());
  bitset2.set(5);
  bitset2.set(32);
  bitset2.set(34);
  bitset1.merge(bitset2);
  res = bitset1.empty();
  EXPECT_EQ(0,res);
  bitones = bitset1.getbitones();
  EXPECT_EQ(3,bitones.size());
  EXPECT_EQ(5,bitones[0]);
  EXPECT_EQ(32,bitones[1]);
  EXPECT_EQ(34,bitones[2]);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
