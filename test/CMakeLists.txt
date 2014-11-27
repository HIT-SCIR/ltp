include_directories (./
    ${SOURCE_DIR}/__util
    ${SOURCE_DIR}/utils
    ${SOURCE_DIR}/__xml4nlp
    ${SOURCE_DIR}/_split_sentence
    ${SOURCE_DIR}/segmentor
    ${SOURCE_DIR}/postagger
    ${SOURCE_DIR}/ner
    ${SOURCE_DIR}/parser
    ${SOURCE_DIR}/srl
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
add_executable (ltp_test ${ltp_test_SRC})
target_link_libraries (ltp_test ${ltp_test_LIBS})

add_executable (cws_cmdline cws_cmdline.cpp)
target_link_libraries (cws_cmdline segmentor_static_lib
    boost_regex_static_lib)
set_target_properties(cws_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (customized_cws_cmdline customized_cws_cmdline.cpp)
target_link_libraries (customized_cws_cmdline segmentor_static_lib
    boost_regex_static_lib)
set_target_properties(customized_cws_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (pos_cmdline pos_cmdline.cpp)
target_link_libraries (pos_cmdline postagger_static_lib)
set_target_properties(pos_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (par_cmdline par_cmdline.cpp)
target_link_libraries (par_cmdline parser_static_lib)
set_target_properties(par_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

# --------------------------------------------------
# multithreaded ltp test is not supported in Windows
# --------------------------------------------------
add_executable (multi_ltp_test ${multi_ltp_test_SRC})
add_executable (multi_cws_cmdline multi_cws_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
set_target_properties(multi_cws_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)
add_executable (multi_customized_cws_cmdline multi_customized_cws_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
set_target_properties(multi_customized_cws_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)
add_executable (multi_pos_cmdline multi_pos_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
set_target_properties(multi_pos_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

if (UNIX)
target_link_libraries (multi_ltp_test ${ltp_test_LIBS} pthread)

target_link_libraries (multi_cws_cmdline segmentor_static_lib
    boost_regex_static_lib
    pthread)

target_link_libraries (multi_customized_cws_cmdline segmentor_static_lib
    boost_regex_static_lib
    pthread)

target_link_libraries (multi_pos_cmdline postagger_static_lib pthread)
else ()
target_link_libraries (multi_ltp_test ${ltp_test_LIBS})

target_link_libraries (multi_cws_cmdline segmentor_static_lib
    boost_regex_static_lib)

target_link_libraries (multi_customized_cws_cmdline segmentor_static_lib
    boost_regex_static_lib)

target_link_libraries (multi_pos_cmdline postagger_static_lib)
endif()

#add_test (NAME ltp_test
#    COMMAND "${CMAKE_COMMAND}"
#    -DTEST_PROG=${EXECUTABLE_TEST_OUTPUT}
#    -P ${CMAKE_MODULE_PATH}/LtpDiff.cmake)
