add_executable(srl_merge_tool merge.cpp process/merge.h)
target_link_libraries(srl_merge_tool dynet srl_srl_static_lib ${LIBS})
set_target_properties (srl_merge_tool PROPERTIES
        OUTPUT_NAME srl_merge_tool
        RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
