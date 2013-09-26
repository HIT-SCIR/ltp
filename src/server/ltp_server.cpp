// Defines the entry point for the Web Service application.
//

#include <sys/wait.h>
#include <unistd.h>             /* For pause() */
#include <stdlib.h>
#include <signal.h>

#include <iostream>

#include "mongoose.h"

#include "Xml4nlp.h"
#include "Ltp.h"

#include "codecs.hpp"
#include "logging.hpp"

#if !defined(LISTENING_PORT)
#define LISTENING_PORT	"12345"
#endif /* !LISTENING_PORT */

#define POST_LEN 1024

using namespace std;
using namespace ltp::strutils::codecs;

static LTP engine;

static int exit_flag;

static int Service(struct mg_connection *conn);

static void signal_handler(int sig_num) {
    exit_flag = sig_num;
}

int main(int argc, char *argv[]) {
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    struct mg_context *ctx;
    const char *options[] = {"listening_ports", LISTENING_PORT, 
        "num_threads", "1", NULL};
    struct mg_callbacks callbacks;

    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.begin_request = Service;

    if ((ctx = mg_start(&callbacks, NULL, options)) == NULL) {
        ERROR_LOG("Cannot initialize Mongoose context");
        exit(EXIT_FAILURE);
    }

    // getchar();
    while (exit_flag == 0) {
        sleep(100000);
    }
    mg_stop(ctx);

    return 0;
}

static int Service(struct mg_connection *conn) {
    char *sentence;
    char type[10];
    char xml[10];
    char buffer[POST_LEN];

    string str_post_data;
    string str_type;
    string str_xml;

    const struct mg_request_info *ri = mg_get_request_info(conn);

    if (!strcmp(ri->uri, "/ltp")) {
        int len;
        while((len = mg_read(conn, buffer, sizeof(buffer) - 1)) > 0){
            buffer[len] = 0;
            str_post_data += buffer;
        }

        TRACE_LOG("CDATA: %s", str_post_data.c_str());
        TRACE_LOG("CDATA length: %d", str_post_data.size());

        sentence = new char[str_post_data.size() + 1];

        mg_get_var(str_post_data.c_str(), 
                str_post_data.size(), 
                "s",
                sentence,
                str_post_data.size());

        mg_get_var(str_post_data.c_str(), 
                str_post_data.size(), 
                "t",
                type,
                sizeof(type) - 1);

        mg_get_var(str_post_data.c_str(), 
                str_post_data.size(), 
                "x",
                xml,
                sizeof(xml) - 1);

        string strSentence = sentence;

        /*
         * validation check
         */
        if (strlen(sentence) == 0 || !isclear(strSentence)) {
            // std::cerr << "Failed validation check" << std::endl;
            WARNING_LOG("Failed string validation check");
            return 0;
        }

        if(strlen(type) == 0) {
            str_type = "";
        } else {
            str_type = type;
        }

        if(strlen(xml) == 0) {
            str_xml = "";
        } else {
            str_xml = xml;
        }

        delete []sentence;

        TRACE_LOG("Input sentence is: %s", strSentence.c_str());

        //Get a XML4NLP instance here.
        XML4NLP    xml4nlp;
        
        if(str_xml == "y"){
            if (-1 == xml4nlp.LoadXMLFromString(strSentence)) {
                // failed the xml validation check
                return 0;
            }

            // move sentence validation check into each module
        } else {
            xml4nlp.CreateDOMFromString(strSentence);
        }

        TRACE_LOG("XML Creation is done.");

        if(str_type == "ws"){
            engine.wordseg(xml4nlp);
        } else if(str_type == "pos"){
            engine.postag(xml4nlp);
        } else if(str_type == "ner"){
            engine.ner(xml4nlp);
        } else if(str_type == "dp"){
            engine.parser(xml4nlp);
        } else if(str_type == "srl"){
            engine.srl(xml4nlp);
        } else {
            engine.srl(xml4nlp);
        }

        TRACE_LOG("Analysis is done.");

        string strResult;
        xml4nlp.SaveDOM(strResult);

        strResult = "HTTP/1.1 200 OK\r\n\r\n" + strResult;

        // cout << "Result is: " << strResult << endl;
        mg_printf(conn, "%s", strResult.c_str());

        xml4nlp.ClearDOM();
    }
    return 1;
}

