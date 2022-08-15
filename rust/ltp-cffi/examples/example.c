//
// Created by 冯云龙 on 2022/8/12.
//

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "ltp.h"

#define MAX_WORD_LEN (10)

struct State {
  char **results;
  size_t *lengths;
};

void store_results(struct State *state, const uint8_t *word, size_t word_len, size_t idx, size_t length) {
  state->results[idx] = malloc(word_len + 1);
  state->lengths[idx] = word_len;

  strncpy(state->results[idx], (const char *) word, word_len);
  state->results[idx][word_len] = '\0';

  if (idx < length - 1) {
    printf("%s ", state->results[idx]);
  } else {
    printf("%s\n", state->results[idx]);
  }

}
int main() {
  const char *cws_model_path = "data/legacy-models/cws_model.bin";
  const char *pos_model_path = "data/legacy-models/pos_model.bin";
  const char *ner_model_path = "data/legacy-models/ner_model.bin";
  Model *cws_model = NULL;
  cws_model = model_load(cws_model_path, strlen(cws_model_path));
  Model *pos_model = NULL;
  pos_model = model_load(pos_model_path, strlen(pos_model_path));
  Model *ner_model = NULL;
  ner_model = model_load(ner_model_path, strlen(ner_model_path));

  const char *sentence = "他叫汤姆去拿外衣";
  size_t word_length[MAX_WORD_LEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t pos_length[MAX_WORD_LEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t ner_length[MAX_WORD_LEN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  char *words[MAX_WORD_LEN] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  char *pos[MAX_WORD_LEN] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  char *ner[MAX_WORD_LEN] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

  struct State word_state = {words, word_length};
  struct State pos_state = {pos, pos_length};
  struct State ner_state = {ner, ner_length};

  Callback cws_callback = {&word_state, store_results};
  size_t length = model_cws_predict(cws_model, sentence, strlen(sentence), cws_callback);

  Callback pos_callback = {&pos_state, store_results};
  model_pos_predict(pos_model, words, word_length, length, pos_callback);

  Callback ner_callback = {&ner_state, store_results};
  model_ner_predict(ner_model, words, word_length, pos, pos_length, length, ner_callback);

  for (size_t i = 0; i < MAX_WORD_LEN; i++) {
    if (words[i] != NULL) { free(words[i]); words[i]=NULL;}
    if (pos[i] != NULL) { free(pos[i]); pos[i]=NULL;}
    if (ner[i] != NULL) { free(ner[i]); ner[i]=NULL;}
  }

  model_release(&cws_model);
  model_release(&pos_model);
  model_release(&ner_model);

  assert(cws_model == NULL);
  assert(pos_model == NULL);
  assert(ner_model == NULL);

  return 0;
}
