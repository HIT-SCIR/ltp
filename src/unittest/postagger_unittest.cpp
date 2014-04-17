#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "postagger/model.h"

const int kMaxTagIndex = 27;
const char * kTags[] = {  "a",  "b",  "c",
  "d",  "e",  "h",  "i",  "j",  "k",  "m",
  "n",  "nd", "nh", "ni", "nl", "ns", "nt",
  "nz", "o",  "p",  "q",  "r",  "u",  "v",
  "wp", "ws", "z"};

TEST(postagger_unittest, test_tags_of_model) {
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());

  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  EXPECT_EQ(kMaxTagIndex, model->labels.size());

  for (int i = 0; i < kMaxTagIndex; ++ i) {
    EXPECT_GE(model->labels.index(kTags[i]), 0);
    EXPECT_LT(model->labels.index(kTags[i]), kMaxTagIndex);
  }

  delete model;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

