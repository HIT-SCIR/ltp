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

link_directories (${LIBRARY_OUTPUT_PATH})

# Training suite is not supported in windows
if (NOT WIN32)
set (SRLExtract_SRC SRLExtract.cpp Corpus.cpp)
add_executable (SRLExtract  ${SRLExtract_SRC})
target_link_libraries (SRLExtract   srl)
set_target_properties (SRLExtract
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

set (SRLGetInstance_SRC SRLGetInstance.cpp GetInstance.cpp)
add_executable (SRLGetInstance ${SRLGetInstance_SRC})
target_link_libraries (SRLGetInstance   srl)
set_target_properties (SRLGetInstance
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

endif()