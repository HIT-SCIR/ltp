include_directories (
    .
    ${util_DIR} 
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/maxent)

set (srl_SRC
    Configuration.cpp
    Configuration.h
    ConstVar.h
    DataPreProcess.cpp
    DataPreProcess.h
    DataStruct.h
    DepSRL.cpp
    DepSRL.h
    FeatureExtractor.cpp
    FeatureExtractor.h
    MyStruct.h
    MyTree.cpp
    MyTree.h
    Sentence.cpp
    Sentence.h
    SRLBaseline.cpp
    SRLBaselineExt.cpp
    SRLBaselineExt.h
    SRLBaseline.h
    SRL_DLL.cpp
    SRL_DLL.h
    SRL_DLL_x.cpp
    tree.hh)

add_library (srl ${srl_SRC})

target_link_libraries (srl util maxent)

configure_file (
    SRL_DLL.h
    ${INCLUDE_OUTPUT_PATH}/ltp/SRL_DLL.h)
