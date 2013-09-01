set (util_DIR            ${SOURCE_DIR}/__util/)
set (utils_DIR           ${SOURCE_DIR}/utils)
set (xml4nlp_DIR         ${SOURCE_DIR}/__xml4nlp/)
set (segmentor_DIR       ${SOURCE_DIR}/segmentor)
set (postagger_DIR       ${SOURCE_DIR}/postagger)
set (parser_DIR          ${SOURCE_DIR}/parser)
set (splitsnt_DIR        ${SOURCE_DIR}/_split_sentence/)
set (ner_DIR             ${SOURCE_DIR}/ner/)
set (srl_DIR             ${SOURCE_DIR}/srl/)
set (ltp_dll_DIR         ${SOURCE_DIR}/__ltp_dll/)
set (server_DIR          ${SOURCE_DIR}/server/)

add_subdirectory ("__util")
add_subdirectory ("__xml4nlp")
add_subdirectory ("_split_sentence")
add_subdirectory ("segmentor")
add_subdirectory ("postagger")
add_subdirectory ("ner")
add_subdirectory ("parser")
add_subdirectory ("srl")
add_subdirectory ("__ltp_dll")

# mongoose server is not supported in windows
if (NOT WIN32)
add_subdirectory ("server")
endif()


