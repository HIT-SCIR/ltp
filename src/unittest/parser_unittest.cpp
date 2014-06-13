#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "parser/model.h"

const int kMaxPostagIndex = 27;
const char * kPostags[] = {"a",  "b",  "c",
  "d",  "e",  "h",  "i",  "j",  "k",  "m",
  "n",  "nd", "nh", "ni", "nl", "ns", "nt",
  "nz", "o",  "p",  "q",  "r",  "u",  "v",
  "wp", "ws", "z"};

const int kMaxDeprelsIndex = 14;
const char * kDeprels[] = {"SBV", "VOB",
  "IOB", "FOB", "DBL", "ATT", "ADV",
  "CMP", "COO", "POB", "LAD", "RAD",
  "IS", "HED",};

TEST(parser_unittest, test_labels_of_model) {
  std::ifstream mfs("./ltp_data/parser.model");
  EXPECT_EQ(true, mfs.is_open());

  ltp::parser::Model * model = new ltp::parser::Model();
  model->load(mfs);

  EXPECT_EQ(kMaxDeprelsIndex, model->num_deprels());
  for (int i = 0; i < kMaxDeprelsIndex; ++ i) {
    EXPECT_GE(model->deprels.index(kDeprels[i]), 0);
    EXPECT_LT(model->deprels.index(kDeprels[i]), kMaxDeprelsIndex);
  }

  EXPECT_EQ(kMaxPostagIndex, model->num_postags());

  for (int i = 0; i < kMaxPostagIndex; ++ i) {
    EXPECT_GE(model->postags.index(kPostags[i]), 0);
    EXPECT_LT(model->postags.index(kPostags[i]), kMaxPostagIndex);
  }
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
