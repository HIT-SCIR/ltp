#include <iostream>
#include <cstring>
#include <ctime>
#include <vector>
#include <list>
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
    Dispatcher( LTP * engine) {
        _engine = engine;
    }

    int next(string &sentence) {
        sentence = "";
        lock_guard<fast_mutex> guard(_mutex);
        if (!getline(cin, sentence, '\n')) {
            return -1;
        }
        return 0;
    }

    void output(const string &result) {
        lock_guard<fast_mutex> guard(_mutex);
        cout << result;
        return;
    }

    LTP* getEngine() {
        return _engine;
    }
    
private:
    fast_mutex     _mutex;
    LTP*          _engine;
};

void multithreaded_ltp( void * args) {
    string sentence;

    Dispatcher * dispatcher = (Dispatcher *)args;
    LTP *  engine = dispatcher->getEngine();

    while (true) {
        int ret = dispatcher->next(sentence);

        if (ret < 0)
            break;
        
        XML4NLP xml4nlp;
        xml4nlp.CreateDOMFromString(sentence);

        if(type == "ws"){
            (*engine).wordseg(xml4nlp);
        } else if(type == "pos"){
            (*engine).postag(xml4nlp);
        } else if(type == "ner"){
            (*engine).ner(xml4nlp);
        } else if(type == "dp"){
            (*engine).parser(xml4nlp);
        } else if(type == "srl"){
            (*engine).srl(xml4nlp);
        } else {
            (*engine).srl(xml4nlp);
        }

        string result;
        xml4nlp.SaveDOM(result);
	dispatcher->output(result);
        xml4nlp.ClearDOM();
        
    }

    return;
}

int main(int argc, char ** argv) {
    
    if (argc != 3) {
        cerr << "Usage: ./ltp_test <config_file> <type>" << endl;
        exit(1);
    }
    
    LTP ltp(argv[1]);
    string _type(argv[2]);
    type = _type;
    Dispatcher * dispatcher = new Dispatcher( &ltp);

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
