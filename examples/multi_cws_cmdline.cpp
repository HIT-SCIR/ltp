/*
 * Multi-threaded segmentor test program. The user input a line
 * of Chinese sentence an the program will output its segment
 * result.
 *
 *  @dependency package: tinythread - a portable c++ wrapper for
 *                       multi-thread library.
 *  @author:             LIU, Yijia
 *  @data:               2013-09-24
 *
 * This program is special designed for UNIX user, for get time
 * is not compilable under MSVC
 */
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <list>
#include <sys/time.h>
#include <sys/types.h>
#include <cstdlib>

#include "segment_dll.h"
#include "tinythread.h"
#include "fast_mutex.h"

using namespace std;
using namespace tthread;

const int MAX_LEN = 1024;

double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

class Dispatcher {
public:
    Dispatcher( void * model ) {
        _model = model;
    }

    int next(string &sentence) {
        sentence = "";
        lock_guard<fast_mutex> guard(_mutex);
        if (!getline(cin, sentence, '\n')) {
            return -1;
        }
        return 0;
    }

    void output(const vector<string> &result) {
        lock_guard<fast_mutex> guard(_mutex);
        for (int i = 0; i < result.size(); ++ i) {
            cout << result[i];
            cout << (i == result.size() - 1 ? '\n' : '|');
        }
        return;
    }

    void * model() {
        return _model;
    }

private:
    fast_mutex      _mutex;
    void *          _model;
    string          _sentence;
};

void multithreaded_segment( void * args) {
    string sentence;
    vector<string> result;

    Dispatcher * dispatcher = (Dispatcher *)args;
    void * model = dispatcher->model();

    while (true) {
        int ret = dispatcher->next(sentence);

        if (ret < 0)
            break;

        result.clear();
        segmentor_segment(model, sentence, result);
        dispatcher->output(result);
    }

    return;
}

int main(int argc, char ** argv) {
    if (argc < 2 || (0 == strcmp(argv[1], "-h"))) {
        std::cerr << "Example: ./multi_cws_cmdline "
                  << "[model path] [lexicon file](optional) threadnum"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "This program recieve input word sequence from stdin." << std::endl;
        std::cerr << "One sentence per line." << std::endl;
        return -1;
    }

    void * engine = 0;
    if (argc == 2) {
        engine = segmentor_create_segmentor(argv[1]);
    } else if (argc == 3) {
        engine = segmentor_create_segmentor(argv[1], argv[2]);
    }

    if (!engine) {
        return -1;
    }
    int num_threads=atoi(argv[3]);
    if(num_threads < 0 || num_threads > thread::hardware_concurrency()) {
        num_threads = thread::hardware_concurrency();
    }
    std::cerr << "TRACE: Model is loaded" << std::endl;
    std::cerr << "TRACE: Running " << num_threads << " thread(s)" << std::endl;

    Dispatcher * dispatcher = new Dispatcher( engine );

    double tm = get_time();
    list<thread *> thread_list;
    for (int i = 0; i < num_threads; ++ i) {
        thread * t = new thread( multithreaded_segment, (void *)dispatcher );
        thread_list.push_back( t );
    }

    for (list<thread *>::iterator i = thread_list.begin();
            i != thread_list.end(); ++ i) {
        thread * t = *i;
        t->join();
        delete t;
    }

    tm = get_time() - tm;
    std::cerr << "TRACE: consume "
        << tm 
        << " seconds."
        << std::endl;


    return 0;
}

