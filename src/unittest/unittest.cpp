#define private   public
#define protected public
#include <iostream>
#include "utils_chartypes_unittest.cpp"
#include "utils_codecs_unittest.cpp"
#include "utils_smartmap_unittest.cpp"
#include "utils_sbcdbc_unittest.cpp"
#include "utils_strpaste_unittest.cpp"
#include "utils_strutils_unittest.cpp"
#include "utils_sentsplit_unittest.cpp"
#include "utils_template_unittest.cpp"
#include "segmentor_unittest.cpp"
#include "parser_unittest.cpp"

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
