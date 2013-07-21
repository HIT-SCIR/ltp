#ifndef __LTP_SEGMENTOR_READER_H__
#define __LTP_SEGMENTOR_READER_H__

#include <iostream>
#include "settings.h"
#include "instance.h"
#include "strutils.hpp"
#include "codecs.hpp"

namespace ltp {
namespace segmentor {

using namespace ltp::strutils;

class SegmentReader {
public:
    SegmentReader(istream & _ifs, int _style = 4) : 
        ifs(_ifs),
        style(_style) {}

    Instance * next() {
        if (ifs.eof()) {
            return 0;
        }

        Instance * inst = new Instance;
        std::string  line;

        std::getline(ifs, line);

        chomp(line);

        if (line.size() == 0) {
            delete inst;
            return 0;
        }

        std::vector<std::string> words = split(line);
        inst->words = words;

        for (int i = 0; i < words.size(); ++ i) {
            std::vector<std::string> chars;
            int num_chars = codecs::decode(words[i], chars);

            // support different style
            if (style == 2) {
                for (int j = 0; j < num_chars; ++ j) {
                    inst->forms.push_back(chars[j]);
                    if (j == 0) {
                        inst->tags.push_back( __b__ );
                    } else {
                        inst->tags.push_back( __i__ );
                    }
                }
            } else if (style == 4) {
                for(int j = 0; j < num_chars; ++ j) {
                    inst->forms.push_back(chars[j]);
                    if (1 == num_chars) {
                        inst->tags.push_back( __s__ );
                    } else {
                        if (0 == j) {
                            inst->tags.push_back( __b__ );
                        } else if (num_chars - 1 == j) {
                            inst->tags.push_back( __e__ );
                        } else {
                            inst->tags.push_back( __i__ );
                        }
                    }
                }
            } else if (style == 6) {
                for (int j = 0; j < num_chars; ++ j) {
                    inst->forms.push_back(chars[j]);

                    if (1 == num_chars) {
                        inst->tags.push_back( __s__ );
                    } else {
                        if (0 == j) {
                            inst->tags.push_back( __b__ );
                        } else if (1 == j) {
                            inst->tags.push_back( __b2__ );
                        } else if (2 == j) {
                            inst->tags.push_back( __b3__ );
                        } else if (num_chars - 1 == j) {
                            inst->tags.push_back( __e__ );
                        } else {
                            inst->tags.push_back( __i__ );
                        }
                    }
                }
            }
        }

        return inst;
    }
private:
    istream &   ifs;
    int         style;
};

}           //  end for namespace segmentor
}           //  end for namespace ltp

#endif      //  end for __LTP_SEGMENTOR_READER_H__
