include_directories (./
    ${SOURCE_DIR}/postagger
    ${SOURCE_DIR}/utils
    ${SOURCE_DIR}/utils/math)

set (postagger_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    postagger.cpp)

add_executable (otpos otpos.cpp ${postagger_SRC})
add_library (postagger postag_dll.cpp ${postagger_SRC})
