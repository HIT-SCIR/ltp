include_directories (./
    ${SOURCE_DIR}/__util
    ${SOURCE_DIR}/__xml4nlp
    ${SOURCE_DIR}/_split_sentence
    ${SOURCE_DIR}/_crfwordsegment
    ${SOURCE_DIR}/_svmtagger
    ${SOURCE_DIR}/_ner
    ${SOURCE_DIR}/_gparser
    ${SOURCE_DIR}/_srl
    ${SOURCE_DIR}/__ltp_dll
    ${THIRDPARTY_DIR}/crfpp
    ${THIRDPARTY_DIR}/maxent
    ${THIRDPARTY_DIR}/tinyxml)

set (ltp_test_SRC           ltp_test.cpp)
set (ltp_test_xml_SRC       ltp_test_xml.cpp)
set (ltp_test2_SRC          ltp_test2.cpp)

add_executable (ltp_test     ${ltp_test_SRC})
add_executable (ltp_test_xml ${ltp_test_xml_SRC})
add_executable (ltp_test2    ${ltp_test2_SRC})

link_directories ( ${LIBRARY_OUTPUT_PATH} )

target_link_libraries (ltp_test
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

target_link_libraries (ltp_test2
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

target_link_libraries (ltp_test_xml
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

#add_test (NAME ltp_test
#    COMMAND "${CMAKE_COMMAND}"
#    -DTEST_PROG=${EXECUTABLE_TEST_OUTPUT}
#    -P ${CMAKE_MODULE_PATH}/LtpDiff.cmake)

