include_directories (
    ./
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/maxent)

set (ner_SRC
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

add_library (ner ${ner_SRC})

configure_file (
    NER_DLL.h
    ${INCLUDE_OUTPUT_PATH}/ltp/NER_DLL.h)
