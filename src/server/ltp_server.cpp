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

#if !defined(LISTENING_PORT)
#define LISTENING_PORT	"12345"
#endif /* !LISTENING_PORT */

#define POST_LEN 1024

using namespace std;

static XML4NLP xml4nlp;
static LTP ltp(xml4nlp);

static int exit_flag;

static int Service(struct mg_connection *conn);

static void signal_handler(int sig_num) {
        exit_flag = sig_num;
}

int main(int argc, char *argv[])
{
        signal(SIGTERM, signal_handler);
        signal(SIGINT, signal_handler);
        struct mg_context *ctx;
        const char *options[] = {"listening_ports", LISTENING_PORT, 
                                 "num_threads", "1", NULL};
        struct mg_callbacks callbacks;
        
        memset(&callbacks, 0, sizeof(callbacks));
        callbacks.begin_request = Service;

        if ((ctx = mg_start(&callbacks, NULL, options)) == NULL) {
                (void) printf("%s\n", "Cannot initialize Mongoose context");
                exit(EXIT_FAILURE);
        }

        // getchar();
        while (exit_flag == 0) {
                sleep(100000);
        }
	mg_stop(ctx);

	return 0;
}

static int Service(struct mg_connection *conn)
{
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
        cout << str_post_data.size() << " " << str_post_data << endl;

        sentence = new char[str_post_data.size() + 1];
        mg_get_var(str_post_data.c_str(), str_post_data.size(), "s", sentence, str_post_data.size());
        mg_get_var(str_post_data.c_str(), str_post_data.size(), "t", type, sizeof(type) - 1);
        mg_get_var(str_post_data.c_str(), str_post_data.size(), "x", xml, sizeof(xml) - 1);

        if (strcmp(sentence, "") == 0)
            return 0;

	if(strcmp(type, "") == 0){
            str_type = "";
	}else{
	    str_type = type;
	}

	if(strcmp(xml, "") == 0){
		str_xml = "";
	} else {
		str_xml = xml;
	}
	
	string strSentence = sentence;
        delete []sentence;

	cout << "Input sentence is: " << strSentence << endl;
	
	if(str_xml == "y"){
		xml4nlp.LoadXMLFromString(strSentence);
	} else {
		xml4nlp.CreateDOMFromString(strSentence);
	}

	if(str_type == "ws"){
		ltp.wordseg();
	} else if(str_type == "pos"){
		ltp.postag();
	} else if(str_type == "ner"){
		ltp.ner();
	} else if(str_type == "dp"){
		ltp.parser();
	} else if(str_type == "srl"){
		ltp.srl();
	} else {
		ltp.srl();
	}

	string strResult;
	xml4nlp.SaveDOM(strResult);
	
	strResult = "HTTP/1.1 200 OK\r\n\r\n" + strResult;

	// cout << "Result is: " << strResult << endl;
        mg_printf(conn, "%s", strResult.c_str());

	xml4nlp.ClearDOM();
    }
    return 1;
}

