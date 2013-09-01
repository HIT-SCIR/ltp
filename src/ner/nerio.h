#ifndef __LTP_NER_IO_H__
#define __LTP_NER_IO_H__

#include <iostream>
#include "settings.h"
#include "instance.h"
#include "strutils.hpp"
#include "sbcdbc.hpp"
#include "codecs.hpp"

namespace ltp {
namespace ner {

using namespace ltp::strutils;

class NERReader {
public:
    NERReader(istream & _ifs, bool _train = false, int _style = 4) : 
        ifs(_ifs),
        train(_train),
        style(_style) {}

    Instance * next() {
        if (ifs.eof()) {
            return 0;
        }

        Instance * inst = new Instance;
        std::string  line;

        std::getline(ifs, line);
        strutils::chomp(line);

        if (line.size() == 0) {
            delete inst;
            return 0;
        }

        std::vector<std::string> words = split(line);
        int found;

        for (int i = 0; i < words.size(); ++ i) {
            if (train) {
                found = words[i].find_last_of('#');
                if (found != std::string::npos) {
                    std::string tag = words[i].substr(found + 1);
                    inst->tags.push_back(tag);
                    words[i] = words[i].substr(0, found);

                    found = words[i].find_last_of('/');
                    if (found != std::string::npos) {
                        std::string postag = words[i].substr(found + 1);
                        inst->postags.push_back(postag);
                        words[i] = words[i].substr(0, found);

                        inst->raw_forms.push_back(words[i]);
                        inst->forms.push_back(strutils::chartypes::sbc2dbc_x(words[i]));
                    } else {
                        delete inst;
                        return 0;
                    }
                } else {
                    delete inst;
                    return 0;
                }
            } else {
                found = words[i].find_last_of('/');
                if (found != std::string::npos) {
                    std::string postag = words[i].substr(found + 1);
                    inst->postags.push_back(postag);
                    words[i] = words[i].substr(0, found);

                    inst->raw_forms.push_back(words[i]);
                    inst->forms.push_back(strutils::chartypes::sbc2dbc_x(words[i]));
                } else {
                    delete inst;
                    return 0;
                }
            }
        }

        return inst;
   }
private:
    istream &   ifs;
    int         style;
    bool        train;
};

class NERWriter {
public:
    NERWriter(std::ostream & _ofs) : ofs(_ofs) {}

    void write(const Instance * inst) {
        int len = inst->size();
        if (inst->predicted_tags.size() != len) {
            return;
        }

        for (int i = 0; i < len; ++ i) {
            ofs << inst->forms[i] 
                << "/" << inst->postags[i]
                << "#" << inst->predicted_tags[i];
            if (i + 1 < len ) {
                ofs << "\t";
            } else {
                ofs << std::endl;
            }
        }
   }

    void debug(const Instance * inst, bool show_feat = false) {
        int len = inst->size();

        for (int i = 0; i < len; ++ i) {
            ofs << inst->forms[i] 
                << "\t" << inst->postags[i]
                << "\t" << inst->tagsidx[i]
                << "\t" << inst->predicted_tagsidx[i]
                << std::endl;
        }
   }
private:
    std::ostream & ofs;
};

}           //  end for namespace ner
}           //  end for namespace ltp
#endif      //  end for __LTP_SEGMENTOR_WRITER_H__
