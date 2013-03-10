MESSAGE(STATUS ${THIRDPARTY})

SET(CMAKE_CXX_FLAGS "-DHAVE_CONFIG_H")
SET(crfwordseg_source
    CRFSegEngine.cpp
    CRFSegEngine.h
    CRFWS.cpp
    CRFWS_DLL.cpp
    CRFWS_DLL.h
    CRFWS_DLL_x.cpp
    CRFWS.h
    CWSTaggerImpl.cpp
    CWSTaggerImpl.h
    DictBase.h
    LASBase.h
    SegEngine.cpp
    SegEngine.h)

INCLUDE_DIRECTORIES(
    ./
    ${THIRDPARTY}/crfpp/
    ${util_dir}/)

ADD_LIBRARY(crfwordseg ${crfwordseg_source})

TARGET_LINK_LIBRARIES(crfwordseg util crfpp)
