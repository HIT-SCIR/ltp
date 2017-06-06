SET(SrlSrlSRC
        config/SrlSrlConfig.h
        model/SrlSrlModel.h
        model/SrlSrlModel.cpp)

add_executable(srl_srl_train train.cpp process/TrainSrlSrl.cpp ${SrlSrlSRC})
target_link_libraries(srl_srl_train dynet ${LIBS})

set_target_properties (srl_srl_train PROPERTIES
        OUTPUT_NAME srl_srl_train
        RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

add_library(srl_srl_static_lib STATIC ${SrlSrlSRC})
target_link_libraries(srl_srl_static_lib dynet ${LIBS})
