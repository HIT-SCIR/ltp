include_directories (./
    ${SOURCE_DIR}/postagger
    ${SOURCE_DIR}/utils
    ${SOURCE_DIR}/math)

set (postagger_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    postagger.cpp
    otpos.cpp)

add_executable (otpos ${postagger_SRC})
