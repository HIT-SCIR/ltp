#define private public
#include <iostream>
#include <sstream>
#include <gtest/gtest.h>
#include "parser.n/io.h"
#include "parser.n/instance.h"

using ltp::depparser::Instance;
using ltp::depparser::CoNLLReader;

TEST(parser_unittest, io_and_instance1) {
  std::stringstream S;
  S << "1 I I PRP PRP _ 2 SUB" << std::endl
    << "2 like like VV VV _ 0 ROOT" << std::endl
    << "3 reading reading NN NN _ 2 VOB" << std::endl;

  CoNLLReader reader(S, true);
  Instance* inst = reader.next();

  EXPECT_EQ(inst->forms.size(), 4);
  EXPECT_EQ(inst->postags.size(), 4);
  EXPECT_EQ(inst->heads.size(), 4);
  EXPECT_EQ(inst->deprels.size(), 4);

  EXPECT_EQ(inst->heads[0], -1);
  EXPECT_EQ(inst->heads[1], 2);
  EXPECT_EQ(inst->heads[2], 0);
  EXPECT_EQ(inst->heads[3], 2);

  EXPECT_TRUE(inst->is_tree());
  EXPECT_TRUE(inst->is_projective());
}

