INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_serialization_VERSION "1.61")

set (boost_serialization_SRC
    src/archive_exception.cpp
    src/basic_archive.cpp
    src/basic_iarchive.cpp
    src/basic_iserializer.cpp
    src/basic_oarchive.cpp
    src/basic_oserializer.cpp
    src/basic_pointer_iserializer.cpp
    src/basic_pointer_oserializer.cpp
    src/basic_serializer_map.cpp
    src/basic_text_iprimitive.cpp
    src/basic_text_oprimitive.cpp
    src/basic_text_wiprimitive.cpp
    src/basic_text_woprimitive.cpp
    src/basic_xml_archive.cpp
    src/basic_xml_grammar.ipp
    src/binary_iarchive.cpp
    src/binary_oarchive.cpp
    src/binary_wiarchive.cpp
    src/binary_woarchive.cpp
    src/codecvt_null.cpp
    src/extended_type_info.cpp
    src/extended_type_info_no_rtti.cpp
    src/extended_type_info_typeid.cpp
    src/polymorphic_iarchive.cpp
    src/polymorphic_oarchive.cpp
    src/singleton.cpp
    src/stl_port.cpp
    src/text_iarchive.cpp
    src/text_oarchive.cpp
    src/text_wiarchive.cpp
    src/text_woarchive.cpp
    src/utf8_codecvt_facet.cpp
    src/void_cast.cpp
    src/xml_archive_exception.cpp
    src/xml_grammar.cpp
    src/xml_iarchive.cpp
    src/xml_oarchive.cpp
    src/xml_wgrammar.cpp
    src/xml_wiarchive.cpp
    src/xml_woarchive.cpp
    )

add_library (boost_serialization_static_lib STATIC ${boost_serialization_SRC})

set_target_properties (boost_serialization_static_lib  PROPERTIES
    OUTPUT_NAME boost_serialization)

add_library (boost_serialization_shared_lib SHARED ${boost_serialization_SRC})

set_target_properties (boost_serialization_shared_lib  PROPERTIES
    VERSION     ${boost_serialization_VERSION}
    OUTPUT_NAME boost_serialization)
