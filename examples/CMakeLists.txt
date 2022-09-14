cmake_minimum_required(VERSION 3.0)
project("LTP - Language Technology Platform")

add_executable(cws_example cws.cpp)
include_directories(cws_example ../include)
target_link_libraries(cws_example segmentor_static_lib)

add_executable(pos_example pos.cpp)
include_directories(pos_example ../include)
target_link_libraries(pos_example postagger_static_lib)

add_executable(ner_example ner.cpp)
include_directories(ner_example ../include)
target_link_libraries(ner_example ner_static_lib)

add_executable(par_example par.cpp)
include_directories(par_example ../include)
target_link_libraries(par_example parser_static_lib)

add_executable(srl_example srl.cpp)
include_directories(srl_example ../include)
target_link_libraries(srl_example srl_static_lib)
