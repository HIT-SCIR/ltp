#include <iostream>
#include <vector>

#include "ltp/postag_dll.h"

int main(int argc, char * argv[]) {
    if (argc < 2) {
        std::cerr << "pos [model path]" << std::endl;
        return -1;
    }

    void * engine = postagger_create_postagger(argv[1]);
    if (!engine) {
        return -1;
    }

    std::vector<std::string> words;

    words.push_back("我");
    words.push_back("是");
    words.push_back("中国人");

    std::vector<std::string> tags;

    postagger_postag(engine, words, tags);

    for (int i = 0; i < tags.size(); ++ i) {
        std::cout << words[i] << "/" << tags[i];
        if (i == tags.size() - 1) std::cout << std::endl;
        else std::cout << " ";

    }

    postagger_release_postagger(engine);
    return 0;
}

