// Defines the entry point for the Web Service application.
//

#include <sys/wait.h>
#include <unistd.h>             /* For pause() */
#include <stdlib.h>

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


static int Service(struct mg_connection *conn);

int main(int argc, char *argv[])
{
        struct mg_context *ctx;
        const char *options[] = {"listening_ports", LISTENING_PORT, NULL};
        struct mg_callbacks callbacks;
        
        memset(&callbacks, 0, sizeof(callbacks));
        callbacks.begin_request = Service;

        if ((ctx = mg_start(&callbacks, NULL, options)) == NULL) {
                (void) printf("%s\n", "Cannot initialize Mongoose context");
                exit(EXIT_FAILURE);
        }

        getchar();

	mg_stop(ctx);

	return 0;
}

static int Service(struct mg_connection *conn)
{
    char sentence[POST_LEN];
    char type[POST_LEN];
    char xml[POST_LEN];
    char post_data[POST_LEN];
    int post_data_len;

    string str_type;
    string str_xml;

    const struct mg_request_info *ri = mg_get_request_info(conn);

    if (!strcmp(ri->uri, "/ltp")) {
        const char *qs = ri->query_string;

        mg_get_var(qs, strlen(qs == NULL ? "" : qs), "s", sentence, sizeof(sentence));
        cout << "sentence: " << sentence << endl;

        mg_get_var(qs, strlen(qs == NULL ? "" : qs), "t", type, sizeof(type));
        mg_get_var(qs, strlen(qs == NULL ? "" : qs), "x", xml, sizeof(xml));
	
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
	int max_len = POST_LEN - 1;
	int split_num = strResult.size() / max_len + 1;
	for(int i = 0; i < split_num; i++){
		mg_printf(conn, "%s", strResult.substr(i * max_len, max_len).c_str());
	}

	xml4nlp.ClearDOM();
    }
    return 1;
}

