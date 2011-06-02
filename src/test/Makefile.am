INCLUDES = -I. 

bin_PROGRAMS = ltp_test ltp_test_xml

ltp_test_SOURCES = ltp_test.cpp

ltp_test_LDADD = \
  -L../_split_sentence -lsplitsnt \
  -L../_crfwordsegment -lcrfwordseg \
  -L../_crfwordsegment/util -lcrfutil \
  -L../_svmtagger -lpostagger \
  -L../_gparser -lgparser \
  -L../_ner -lner \
  -L../_srl -lsrl \
  -L../__xml4nlp -lxml4nlp \
  -L../__maxent -lme \
  -L../__ltp_dll -lltp \
  -L../__crf++ -lcrfpp \
  -L../__util -lutil 

ltp_test_LDFLAGS = \
  -ldl -lpthread -lboost_regex

ltp_test_xml_SOURCES = ltp_test_xml.cpp

ltp_test_xml_LDADD = \
  -L../_split_sentence -lsplitsnt \
  -L../_crfwordsegment -lcrfwordseg \
  -L../_crfwordsegment/util -lcrfutil \
  -L../_svmtagger -lpostagger \
  -L../_gparser -lgparser \
  -L../_ner -lner \
  -L../_srl -lsrl \
  -L../__xml4nlp -lxml4nlp \
  -L../__maxent -lme \
  -L../__ltp_dll -lltp \
  -L../__crf++ -lcrfpp \
  -L../__util -lutil 

ltp_test_xml_LDFLAGS = \
  -ldl -lpthread -lboost_regex

