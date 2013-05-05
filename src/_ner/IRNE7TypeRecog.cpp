#include "IRNE7TypeRecog.h"
#include <math.h>

#define MAX 2048

IRNErecog::IRNErecog()
{
}

IRNErecog::~IRNErecog()
{
}

void IRNErecog::crf_set(char *model_path)
{
    char cmd[256];
    strcpy(cmd, "-m ");
    strcat(cmd, model_path);
    strcat(cmd, "crf_gbk.model ");
    strcat(cmd, "-n1 ");

    this->tagger = CRFPP::createTagger(cmd);

    if (!this->tagger)
	fprintf(stderr, "crf_set: create failed.\n");
}



void IRNErecog::IRNE7TypeRecog(const string& strSen, string& StrOut, int tagForm, bool* isNEtypeFlag)
{
    register unsigned int i, j;
    unsigned int k;
    register const char *crf_result; 
    const char * from;
    char to[MAX], ltp_result[1024];
    enum machine_state {
	STATE_A, STATE_B, 
	STATE_C, STATE_D, 
	STATE_E, STATE_F 
    } state;


    i = j = 0;
    from = strSen.c_str();
    state = STATE_A;

    /************
     * from[i]; *
     * to[j];   *
     ************/

    /* convert from input */
    while (from[i]) {
	switch(state) {
	case STATE_A:
	    /* do something */
	    /* next state */
	    if (from[i] < 0)
		state = STATE_B;
	    else if (from[i] == '/')
		state = STATE_D;
	    else if (from[i] == ' ' ||
		     from[i] == '\t' ||
		     from[i] == '\n' ||
		     from[i] == '\r')
		state = STATE_E;
	    else 
		state = STATE_F;
	    break;
	case STATE_B:
	    /* do something */
	    to[j++] = from[i++];
	    /* next state */
	    state = STATE_C;
	    break;
	case STATE_C:
	    /* do something */
	    to[j++] = from[i++];
	    /* next state */
	    state = STATE_A;
	    break;
	case STATE_D:
	    /* do something */
	    i++;
	    to[j++] = '\t';
	    /* next state */
	    state = STATE_A;
	    break;
	case STATE_E:
	    /* do something */
	    i++;
	    to[j++] = '\0';
	    this->tagger->add(
		strncpy(
		    (char *) malloc(
			j * sizeof(char)), 
		    to, 
		    j));
	    /* I mean this:
	    {
		char *str = (char *) malloc(j * sizeof(*str));
		strncpy(str, to, j);
		this->tagger->add(str);
	    }
	    without need of extra variable ``str'' */
	    j = 0;
	    /* next state */
	    state = STATE_A;
	    break;
	case STATE_F:
	    /* do something */
	    to[j++] = from[i++];
	    /* next state */
	    state = STATE_A;
	    break;
	default:
	    fprintf(stderr, "error in IRNE7TypeRecog: no such state.\n");
	    break;
	}
    }

    /* crf++ parse */
    if (!this->tagger->parse())
	fprintf(stderr, "error in IRNE7TypeRecog: crf++ parse failed");

    /* debug: print results from crf++
    for (i = 0; i < this->tagger->size(); ++i) {
	for (j = 0; j < this->tagger->xsize(); ++j)
	    fprintf(stderr, "%s\t", this->tagger->x(i, j));
	fprintf(stderr, "%s\n", this->tagger->y2(i));
    }
    */

    /* convert to output */
    i = j = 0;
    for (k = 0; k < this->tagger->size(); ++k) {
	switch (*(this->tagger->y2(k))) {
	case 'O':
	    /* print x(k, 0) */
	    crf_result = this->tagger->x(k, 0);
	    for (i = 0; crf_result[i]; i += 2) {
		ltp_result[j++] = crf_result[i];
		ltp_result[j++] = crf_result[i + 1];
	    }
	    /* print '/' */
	    ltp_result[j++] = '/'; 
	    /* print x(k, 1) */
	    crf_result = this->tagger->x(k, 1);
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print '#' */
	    ltp_result[j++] = '#';
	    /* print 'O' */
	    ltp_result[j++] = 'O';
	    /* print ' ' */
	    ltp_result[j++] = ' ';
	    break;
	case 'S':
	    /* print x(k, 0) */
	    crf_result = this->tagger->x(k, 0);
	    for (i = 0; crf_result[i]; i += 2) {
		ltp_result[j++] = crf_result[i];
		ltp_result[j++] = crf_result[i + 1];
	    }
	    /* print '/' */
	    ltp_result[j++] = '/'; 
	    /* print x(k, 1) */
	    crf_result = this->tagger->x(k, 1);
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print '#S-' */
	    ltp_result[j++] = '#'; 
	    ltp_result[j++] = 'S'; 
	    ltp_result[j++] = '-'; 
	    /* print y2(k) */
	    crf_result = this->tagger->y2(k) + 2;
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print ' ' */
	    ltp_result[j++] = ' ';
	    break;
	case 'B':
	    /* print x(k, 0) */
	    crf_result = this->tagger->x(k, 0);
	    for (i = 0; crf_result[i]; i += 2) {
		ltp_result[j++] = crf_result[i];
		ltp_result[j++] = crf_result[i + 1];
	    }
	    /* print '\' */
	    ltp_result[j++] = '/'; 
	    /* print x(k, 1) */
	    crf_result = this->tagger->x(k, 1);
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print '#B-' */
	    ltp_result[j++] = '#';
	    ltp_result[j++] = 'B';
	    ltp_result[j++] = '-';
	    /* print y2(k) */
	    crf_result = this->tagger->y2(k) + 2;
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print ' ' */
	    ltp_result[j++] = ' ';
	    break;
	case 'I':
	    /* print x(k, 0) */
	    crf_result = this->tagger->x(k, 0);
	    for (i = 0; crf_result[i]; i += 2) {
		ltp_result[j++] = crf_result[i];
		ltp_result[j++] = crf_result[i + 1];
	    }
	    /* print '\' */
	    ltp_result[j++] = '/'; 
	    /* print x(k, 1) */
	    crf_result = this->tagger->x(k, 1);
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print '#I-' */
	    ltp_result[j++] = '#';
	    ltp_result[j++] = 'I';
	    ltp_result[j++] = '-';
	    /* print y2(k) */
	    crf_result = this->tagger->y2(k) + 2;
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print ' ' */
	    ltp_result[j++] = ' ';
	    break;
	case 'E':
	    /* print x(k, 0) */
	    crf_result = this->tagger->x(k, 0);
	    for (i = 0; crf_result[i]; i += 2) {
		ltp_result[j++] = crf_result[i];
		ltp_result[j++] = crf_result[i + 1];
	    }
	    /* print '\' */
	    ltp_result[j++] = '/'; 
	    /* print x(k, 1) */
	    crf_result = this->tagger->x(k, 1);
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print '#E-' */
	    ltp_result[j++] = '#';
	    ltp_result[j++] = 'E';
	    ltp_result[j++] = '-';
	    /* print y2(k) */
	    crf_result = this->tagger->y2(k) + 2;
	    for (i = 0; crf_result[i]; ++i)
		ltp_result[j++] = crf_result[i];
	    /* print ' ' */
	    ltp_result[j++] = ' ';
	    break;
	default:
	    fprintf(stderr,
		    "error in IRNE7TypeRecog: no such tag %c\n",
		    *(this->tagger->y2(k)));
	    break;
	}
    }
    ltp_result[j++] = '\0';

    /* debug: print result in char 
       fprintf(stderr, "IRNE7TypeRecog: result is %s\n", ltp_result); */

    StrOut.assign(ltp_result);
    /* debug: print result in string 
       fprintf(stdout, "final result: %s\n", StrOut.c_str()); */
}
