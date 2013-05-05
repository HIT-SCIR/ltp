include_directories (
    ./
#    ${THIRDPARTY_DIR}/boost/include
#    ${THIRDPARTY_DIR}/maxent
    ${THIRDPARTY_DIR}/crfpp/
    ${util_DIR}/)

set (ner_SRC
#    Dictionary.cpp
#    Dictionary.h
#    InitDic.cpp
#    InitDic.h
#    Model.cpp
#    Model.h
#    MyNLPlib.cpp
#    MyNLPlib.h
#    NBestHMM.cpp
#    NBestHMM.h
#    RuleNErecog.cpp
#    RuleNErecog.h
    IRNE7TypeRecog.cpp
    IRNE7TypeRecog.h
    NER_DLL.cpp
    NER_DLL.h
    NER_DLL_x.cpp)

add_library (ner ${ner_SRC})

target_link_libraries (ner util crfpp)

configure_file (
    NER_DLL.h
    ${INCLUDE_OUTPUT_PATH}/ltp/NER_DLL.h)
