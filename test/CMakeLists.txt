include_directories (./
    ${SOURCE_DIR}/__util
    ${SOURCE_DIR}/__xml4nlp
    ${SOURCE_DIR}/_split_sentence
    ${SOURCE_DIR}/_ner
    ${SOURCE_DIR}/_srl
    ${SOURCE_DIR}/__ltp_dll
    ${THIRDPARTY_DIR}/tinythreadpp
    ${THIRDPARTY_DIR}/crfpp
    ${THIRDPARTY_DIR}/maxent
    ${THIRDPARTY_DIR}/tinyxml)

set (ltp_test_SRC           ltp_test.cpp)
set (multi_ltp_test_SRC     multi_ltp_test.cpp 
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
#set (ltp_test_xml_SRC       ltp_test_xml.cpp)
#set (ltp_test2_SRC          ltp_test2.cpp)

link_directories ( ${LIBRARY_OUTPUT_PATH} )

add_executable (ltp_test ${ltp_test_SRC})
#add_executable (ltp_test_xml ${ltp_test_xml_SRC})
#add_executable (ltp_test2    ${ltp_test2_SRC})

set (ltp_test_LIBS 
maxent_static_lib
util_static_lib
ltp
splitsnt_static_lib
segmentor_static_lib
postagger_static_lib
parser_static_lib
ner_static_lib
srl_static_lib
xml4nlp
boost_regex_static_lib)

if (NOT WIN32) 
set (ltp_test_LIBS ${ltp_test_LIBS} pthread)
endif()

target_link_libraries (ltp_test ${ltp_test_LIBS})

#
# multithreaded ltp test is not supported in Windows
# 
if (NOT WIN32) 
add_executable (multi_ltp_test ${multi_ltp_test_SRC})
target_link_libraries (multi_ltp_test ${ltp_test_LIBS} pthread)
endif()

#target_link_libraries (ltp_test2
#    maxent
#    crfpp
#    util
#    ltp
#    splitsnt
#    crfwordseg
#    postagger
#    gparser
#    ner
#    srl
#    xml4nlp
#    dl
#    pthread
#    boost_regex)

#target_link_libraries (ltp_test_xml
#    maxent
#    crfpp
#    util
#    ltp
#    splitsnt
#    crfwordseg
#    postagger
#    gparser
#    ner
#    srl
#    xml4nlp
#    dl
#    pthread
#    boost_regex)

#add_test (NAME ltp_test
#    COMMAND "${CMAKE_COMMAND}"
#    -DTEST_PROG=${EXECUTABLE_TEST_OUTPUT}
#    -P ${CMAKE_MODULE_PATH}/LtpDiff.cmake)
