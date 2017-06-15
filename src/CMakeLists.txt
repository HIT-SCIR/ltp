set (utils_DIR          ${SOURCE_DIR}/utils)
set (xml4nlp_DIR        ${SOURCE_DIR}/xml4nlp/)
set (segmentor_DIR      ${SOURCE_DIR}/segmentor)
set (postagger_DIR      ${SOURCE_DIR}/postagger)
set (parser_DIR         ${SOURCE_DIR}/parser)
set (parser_n_DIR       ${SOURCE_DIR}/parser_n)
set (splitsnt_DIR       ${SOURCE_DIR}/splitsnt)
set (ner_DIR            ${SOURCE_DIR}/ner/)
set (srl_DIR            ${SOURCE_DIR}/srl/)
set (ltp_DIR            ${SOURCE_DIR}/ltp/)
set (server_DIR         ${SOURCE_DIR}/server/)


add_subdirectory ("xml4nlp")
add_subdirectory ("splitsnt")
add_subdirectory ("segmentor")
add_subdirectory ("postagger")
add_subdirectory ("ner")
#add_subdirectory ("parser")
add_subdirectory ("parser.n")
add_subdirectory ("srl")
add_subdirectory ("ltp")
add_subdirectory ("console")

# mongoose server is not supported in windows
if (UNIX)
    add_subdirectory ("server")
    # only compile unittest with gnu-c++<5.0
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
            add_subdirectory ("unittest")
        endif()
    endif()
endif()


