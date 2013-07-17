#ifndef __LTP_POSTAGGER_IO_H__
#define __LTP_POSTAGGER_IO_H__

#include <iostream>
#include "settings.h"
#include "instance.h"
#include "strutils.hpp"

namespace ltp {
namespace postagger {

class PostaggerReader {
public:
    PostaggerReader(istream & _ifs, bool _train = false) : 
        ifs(_ifs),
        train(_train) {}

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

        for (int i = 0; i < words.size(); ++ i) {
            if (train) {
                std::vector<std::string> sep = strutils::rsplit_by_sep(words[i], "_", 1);
                if (sep.size() == 2) {
                    inst->forms.push_back(sep[0]);
                    inst->tags.push_back(sep[1]);
                } else {
                    delete inst;
                    return 0;
                }
            } else {
                inst->forms.push_back(words[i]);
            }
        }

        return inst;
    }
private:
    istream &   ifs;
    int         train;
};

/*
 * postag writer class, use to write postag to a stream.
 */
class PostaggerWriter {
public:
    PostaggerWriter(std::ostream & _ofs) : ofs(_ofs) {}

    void write(const Instance * inst) {

    }

    void debug(const Instance * inst, bool show_feat = false) {
    }
private:
    std::ostream & ofs;
};

}           //  end for namespace postagger
}           //  end for namespace ltp
#endif      //  end for __LTP_POSTAGGER_IO_H__
