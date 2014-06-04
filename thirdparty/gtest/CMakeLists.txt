include_directories (./ ./src/ ./include)
set (libgtest_SOURCE src/gtest-all.cc)
add_library (gtest ${libgtest_SOURCE})
