#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "postagger/model.h"
#include "postagger/constrainutil.hpp"

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

TEST(postagger_unittest, test_load_constrain) {
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-single.constrain");
  EXPECT_EQ(nr_constraints, 1);
}

TEST(postagger_unittest, test_load_constrain_to_uninitialized_model) {
  ltp::postagger::Model * model = new ltp::postagger::Model();
  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-single.constrain");
  EXPECT_EQ(nr_constraints, 0);
}

TEST(postagger_unittest, test_load_constrain_with_unknown_tag) {
  // It's expected to show a warning log complain that the tag
  // was not known.
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-unknown.constrain");
  EXPECT_EQ(nr_constraints, 0);
}

TEST(postagger_unittest, test_load_constrain_with_known_and_unknown_tag) {
  // There are two tag in the constrain file (one is known and another one
  // unknown). One constrain should be loaded, but only one bit should be
  // activated.
  std::ifstream mfs("./ltp_data/pos.model");
  EXPECT_EQ(true, mfs.is_open());
  ltp::postagger::Model * model = new ltp::postagger::Model();
  model->load(mfs);

  int nr_constraints = load_constrain(model,
      "./test_data/unittest/postag-known-unknown.constrain");
  EXPECT_EQ(nr_constraints, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

