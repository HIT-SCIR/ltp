add_subdirectory (boost)
add_subdirectory (maxent)
add_subdirectory (tinyxml)
#add_subdirectory (cnn)
add_subdirectory (jsoncpp)

if (UNIX)
add_subdirectory (gtest)
endif()

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cnn)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.2.10)

#set(WITH_EIGEN_BACKEND 1)
