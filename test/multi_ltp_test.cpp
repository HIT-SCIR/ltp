/*
 * this is the multi-threaded test suite for LTP.
 *
 *  @author: NIU, Guochen <gcniu@ir.hit.edu.cn>
 *           HAN, Bin     <bhan@ir.hit.edu.cn>
 *           LIU, Yijia   <yjliu@ir.hit.edu.cn>
 *
 *  @data:  2013-09-26
 */
#include <iostream>
#include <cstring>
#include <ctime>
#include <vector>
#include <list>
#include <map>
#include <sys/time.h>
#include <sys/types.h>

#include "tinythread.h"
#include "fast_mutex.h"

#include "Xml4nlp.h"
#include "Ltp.h"

using namespace std;
using namespace tthread;

string type;

double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

class Dispatcher {
public:
    Dispatcher( LTP * engine, std::istream & ifs) : 
        _engine(engine),
        _max_idx(0), 
        _idx(0),
        _ifs(ifs) {}

    int next(string &sentence) {
        sentence = "";
        lock_guard<fast_mutex> guard(_mutex);
        if (!getline(_ifs, sentence, '\n')) {
            return -1;
        } 
        return _max_idx ++;
    }

    void output(int idx, const string &result) {
        lock_guard<fast_mutex> guard(_mutex);

        if (idx > _idx) {
            _back[idx] = result;
        } else if (idx == _idx) {
            std::cout << result << std::endl;
            ++ _idx;

            std::map<int, std::string>::iterator itx;
            itx = _back.find(_idx);

            while (itx != _back.end()) {
                std::cout << itx->second << std::endl;

                _back.erase(itx);
                ++ _idx;
                itx = _back.find(_idx);
            }
        }
        return;
    }

    LTP* get_engine() {
        return _engine;
    }

private:
    fast_mutex  _mutex;
    LTP *       _engine;
    int         _max_idx;
    int         _idx;

    std::istream & _ifs;
    std::map<int, std::string> _back;
};

void multithreaded_ltp( void * args) {
    string sentence;

    Dispatcher * dispatcher = (Dispatcher *)args;
    LTP *  engine = dispatcher->get_engine();

    while (true) {
        int ret = dispatcher->next(sentence);

        if (ret < 0)
            break;

        XML4NLP xml4nlp;
        xml4nlp.CreateDOMFromString(sentence);

        if (type == "sp"){
            engine->splitSentence_dummy(xml4nlp);
        } else if(type == "ws"){
            engine->wordseg(xml4nlp);
        } else if(type == "pos"){
            engine->postag(xml4nlp);
        } else if(type == "ner"){
            engine->ner(xml4nlp);
        } else if(type == "dp"){
            engine->parser(xml4nlp);
        } else if(type == "srl"){
            engine->srl(xml4nlp);
        } else {
            engine->srl(xml4nlp);
        }

        string result;
        // for segmentation only

        if (type == "sp")
        {
            vector<string> sents;
            xml4nlp.GetSentencesFromParagraph(sents, 0);
            size_t jj = 0;
            for (; jj < sents.size() - 1; ++jj)
            {
                result += sents[jj];
                result += "\n";
            }
            result += sents[jj];
        }

        if (type == "ws")
        {
            vector<string> words;
            xml4nlp.GetWordsFromSentence(words, 0);
            size_t ii = 0;
            for (; ii < words.size()-1; ++ii)
                result += words[ii] + " ";
            result += words[ii];
        }

        /*
        xml4nlp.SaveDOM(result);
        xml4nlp.ClearDOM();
        */

        dispatcher->output(ret, result);
    }

    return;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        cerr << "Usage: ./ltp_test <config_file> <type> <test_file>" << endl;
        exit(1);
    }

    LTP engine(argv[1]);
    if (!engine.loaded()) {
        std::cerr << "Failed to load LTP" << std::endl;
        return -1;
    }

    std::string _type(argv[2]);
    std::ifstream ifs(argv[3]);
    type = _type;
    Dispatcher * dispatcher = new Dispatcher( &engine, ifs);

    int num_threads = thread::hardware_concurrency();
    std::cerr << "TRACE: LTP is built" << std::endl;
    std::cerr << "TRACE: Running " << num_threads << " thread(s)" << std::endl;

    double tm = get_time();
    list<thread *> thread_list;
    for (int i = 0; i < num_threads; ++ i) {
        thread * t = new thread(multithreaded_ltp, (void *)dispatcher );
        thread_list.push_back( t );
    }

    for (list<thread *>::iterator i = thread_list.begin();
        i != thread_list.end(); ++ i) {
        thread * t = *i;
        t->join();
        delete t;
    }

    delete dispatcher;
    tm = get_time() - tm;
    std::cerr << "TRACE: consume "
        << tm 
        << " seconds."
        << std::endl;

    return 0;
}
