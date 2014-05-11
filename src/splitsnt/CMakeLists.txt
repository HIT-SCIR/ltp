include_directories (${SOURCE_DIR})

set (splitsnt_VERSION "0.0.1")

add_library (splitsnt_static_lib SplitSentence.cpp)

set_target_properties (splitsnt_static_lib  PROPERTIES
    OUTPUT_NAME     splitsnt)

add_library (splitsnt_shared_lib SHARED SplitSentence.cpp)

set_target_properties (splitsnt_shared_lib    PROPERTIES
    VERSION         ${splitsnt_VERSION}
    OUTPUT_NAME     splitsnt)


