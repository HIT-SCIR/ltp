INCLUDE_DIRECTORIES(
    ./
    ${THIRDPARTY}/boost/include
    ${THIRDPARTY}/maxent)

SET(ner_source
    Dictionary.cpp
    Dictionary.h
    InitDic.cpp
    InitDic.h
    IRNE7TypeRecog.cpp
    IRNE7TypeRecog.h
    Model.cpp
    Model.h
    MyNLPlib.cpp
    MyNLPlib.h
    NBestHMM.cpp
    NBestHMM.h
    NER_DLL.cpp
    NER_DLL.h
    NER_DLL_x.cpp
    RuleNErecog.cpp
    RuleNErecog.h)

ADD_LIBRARY(ner ${ner_source})
