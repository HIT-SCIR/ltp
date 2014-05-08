#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "segmentor/model.h"

const int kMaxTagIndex = 4;
const char * kTags[] = {"b", "i", "e", "s"};

TEST(segmentor_unittest, test_tags_of_model) {
  std::ifstream mfs("./ltp_data/cws.model");
  EXPECT_EQ(true, mfs.is_open());

  ltp::segmentor::Model * model = new ltp::segmentor::Model();
  model->load(mfs);

  EXPECT_EQ(kMaxTagIndex, model->labels.size());
  for (int i = 0; i < kMaxTagIndex; ++ i) {
    EXPECT_GE(model->labels.index(kTags[i]), 0);
    EXPECT_LT(model->labels.index(kTags[i]), kMaxTagIndex);
  }
  delete model;
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
