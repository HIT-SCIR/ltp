INCLUDE_DIRECTORIES(./
    ${util_dir}
    ${xml4nlp_dir}
    ${splitsnt_dir}
    ${crfwordseg_dir}
    ${svmtagger_dir}
    ${ner_dir}
    ${gparser_dir}
    ${srl_dir}
    ${ltp_dll_dir}
    ${THIRDPARTY}/crfpp
    ${THIRDPARTY}/maxent
    ${THIRDPARTY}/tinyxml)

SET(ltp_test_source     ltp_test.cpp)
SET(ltp_test_xml_source ltp_test_xml.cpp)

ADD_EXECUTABLE(ltp_test     ${ltp_test_source})
ADD_EXECUTABLE(ltp_test_xml ${ltp_test_xml_source})

TARGET_LINK_LIBRARIES(ltp_test
    maxent
    crfpp
    util
    ltp
    splitsnt
    crfwordseg
    postagger
    gparser
    ner
    srl
    xml4nlp
    dl
    pthread
    boost_regex)

TARGET_LINK_LIBRARIES(ltp_test_xml
    maxent
    crfpp
    util
    ltp
    splitsnt
    crfwordseg
    postagger
    gparser
    ner
    srl
    xml4nlp
    dl
    pthread
    boost_regex)
