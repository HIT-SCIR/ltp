set (CMAKE_CXX_FLAGS "-DHAVE_CONFIG_H")
set (crfwordseg_source
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

include_directories (
    ./
    ${THIRDPARTY_DIR}/crfpp/
    ${util_DIR}/)

add_library (crfwordseg ${crfwordseg_source})

target_link_libraries (crfwordseg util crfpp)

configure_file (
    CRFWS_DLL.h 
    ${INCLUDE_OUTPUT_PATH}/ltp/CRFWS_DLL.h
    @ONLY)
