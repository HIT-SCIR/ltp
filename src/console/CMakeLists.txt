include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/boost/include/
    ${THIRDPARTY_DIR}/tinythreadpp
    ${THIRDPARTY_DIR}/maxent
    ${THIRDPARTY_DIR}/tinyxml
    ${THIRDPARTY_DIR}/jsoncpp/include)

set (ltp_test_SRC ltp_test.cpp ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)

link_directories ( ${LIBRARY_OUTPUT_PATH} )
add_executable (ltp_test ${ltp_test_SRC})
target_link_libraries (ltp_test
    maxent_static_lib
    ltp
    splitsnt_static_lib
    segmentor_static_lib
    postagger_static_lib
    parser_static_lib
    ner_static_lib
    srl_static_lib
    xml4nlp
    boost_regex_static_lib
    boost_program_options_static_lib
    jsoncpp)

add_executable (cws_cmdline cws_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
target_link_libraries (cws_cmdline segmentor_static_lib
    boost_program_options_static_lib
    boost_regex_static_lib)
set_target_properties (cws_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (pos_cmdline pos_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
target_link_libraries (pos_cmdline postagger_static_lib
    boost_program_options_static_lib)
set_target_properties (pos_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (par_cmdline par_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
target_link_libraries (par_cmdline parser_static_lib
    boost_program_options_static_lib)
set_target_properties (par_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)

add_executable (ner_cmdline ner_cmdline.cpp
    ${THIRDPARTY_DIR}/tinythreadpp/tinythread.cpp)
target_link_libraries (ner_cmdline ner_static_lib
    boost_program_options_static_lib)
set_target_properties (ner_cmdline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/examples/)


if (NOT MSVC AND NOT MINGW)
  target_link_libraries (ltp_test pthread)
  target_link_libraries (cws_cmdline pthread)
  target_link_libraries (pos_cmdline pthread)
  target_link_libraries (par_cmdline pthread)
  target_link_libraries (ner_cmdline pthread)
endif ()


