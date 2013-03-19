INCLUDE_DIRECTORIES(. ${util_dir} ${THIRDPARTY}/boost/include ${THIRDPARTY}/maxent)

SET(srl_source
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

ADD_LIBRARY(srl ${srl_source})

TARGET_LINK_LIBRARIES(srl util maxent)
