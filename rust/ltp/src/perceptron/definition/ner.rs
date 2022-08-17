use crate::perceptron::definition::GenericItem;
use crate::perceptron::{Definition, Sample};
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct NERDefinition {
    to_labels: Vec<String>,
    labels_to: HashMap<String, usize>,
}

impl NERDefinition {
    pub fn new(to_labels: Vec<String>) -> Self {
        let labels_to = to_labels
            .iter()
            .enumerate()
            .map(|(i, label)| (label.clone(), i))
            .collect();
        NERDefinition {
            labels_to,
            to_labels,
        }
    }

    /// +----------------+-----------------------------------------------------------+
    // | 类别           | 特征                                                        |
    // +================+============================================================+
    // | word-unigram   | w[-2], w[-1], w[0], w[1], w[2]                             |
    // +----------------+------------------------------------------------------------+
    // | word-bigram    | w[-2]w[-1],w[-1]w[0],w[0]w[1],w[1]w[2],w[-2]w[0],w[0]w[2]  |
    // +----------------+------------------------------------------------------------+
    // | postag-unigram | p[-2],p[-1],p[0],p[1],p[2]                                 |
    // +----------------+------------------------------------------------------------+
    // | postag-bigram  | p[-1]p[0],p[0]p[1]                                         |
    // +----------------+------------------------------------------------------------+
    pub fn parse_words_features(&self, words: &[&str], poses: &[&str]) -> Vec<Vec<String>> {
        let word_null = "";
        let words_len = words.len();
        let mut features = Vec::with_capacity(words_len);

        for (idx, &cur_word) in words.iter().enumerate() {
            // 剩余字符数
            let last = words_len - idx - 1;
            let pre2_word = if idx > 1 { words[idx - 2] } else { word_null };
            let pre_word = if idx > 0 { words[idx - 1] } else { word_null };
            let next_word = if last > 0 { words[idx + 1] } else { word_null };
            let next2_word = if last > 1 { words[idx + 2] } else { word_null };

            // todo: 优化容量设置
            let mut feature = Vec::with_capacity(18);

            // w[0]
            feature.push(format!("2{}", words[idx]));

            // p[0]
            feature.push(format!("d{}", poses[idx]));

            if idx > 0 {
                feature.push(format!("1{}", pre_word)); // w[-1]
                feature.push(format!("6{}{}", pre_word, cur_word)); // w[-1]w[0]
                feature.push(format!("c{}", poses[idx - 1])); // p[-1]
                feature.push(format!("g{}{}", poses[idx - 1], poses[idx])); // p[-1]p[0]
                if idx > 1 {
                    feature.push(format!("0{}", pre2_word)); // w[-2]
                    feature.push(format!("5{}{}", pre2_word, pre_word)); // w[-2]w[-1]
                    feature.push(format!("9{}{}", pre2_word, cur_word)); // w[-2]w[0]

                    feature.push(format!("b{}", poses[idx - 2])); // p[-2]
                }
            }

            if last > 0 {
                feature.push(format!("3{}", next_word)); // w[+1]
                feature.push(format!("7{}{}", cur_word, next_word)); // w[0]w[+1]
                feature.push(format!("e{}", poses[idx + 1])); // p[+1]
                feature.push(format!("h{}{}", poses[idx], poses[idx + 1])); // p[0]p[+1]
                if last > 1 {
                    feature.push(format!("4{}", next2_word)); // w[+2]
                    feature.push(format!("8{}{}", next_word, next2_word)); // w[+1]w[+2]
                    feature.push(format!("a{}{}", cur_word, next2_word)); // w[0]w[+2]
                    feature.push(format!("f{}", poses[idx + 2])); // p[+2]
                }
            }

            features.push(feature);
        }
        features
    }
}

impl Definition for NERDefinition {
    type Fragment = dyn for<'any> GenericItem<'any, Item = ()>;
    type Prediction = dyn for<'any> GenericItem<'any, Item = Vec<&'any str>>;
    type RawFeature =
        dyn for<'any> GenericItem<'any, Item = (&'any [&'any str], &'any [&'any str])>;

    fn use_viterbi(&self) -> bool {
        true
    }

    fn labels(&self) -> Vec<String> {
        self.to_labels.clone()
    }

    fn label_num(&self) -> usize {
        self.to_labels.len()
    }

    fn label_to(&self, label: &str) -> usize {
        self.labels_to[label]
    }

    fn to_label(&self, index: usize) -> &str {
        &self.to_labels[index]
    }

    fn parse_features(
        &self,
        line: &<Self::RawFeature as GenericItem>::Item,
    ) -> ((), Vec<Vec<String>>) {
        let (words, features) = line;
        let features = self.parse_words_features(words, features);

        ((), features)
    }

    #[cfg(feature = "parallel")]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Vec<Sample> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();
        let mut result = Vec::with_capacity(lines.len());

        lines
            .par_iter()
            .map(|sentence| {
                let words_tags = sentence.split_whitespace().collect_vec();

                let mut words = Vec::with_capacity(words_tags.len());
                let mut poses = Vec::with_capacity(words_tags.len());
                let mut labels = Vec::with_capacity(words_tags.len());
                for word_tag in words_tags {
                    let result = word_tag.rsplitn(3, '/');
                    let (label, pos, word) = result.collect_tuple().expect("tag not found");
                    words.push(word);
                    poses.push(pos);
                    labels.push(self.label_to(label));
                }
                let features = self.parse_words_features(&words, &poses);
                (features, labels)
            })
            .collect_into_vec(&mut result);

        result
    }

    #[cfg(not(feature = "parallel"))]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Vec<Sample> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();

        lines
            .iter()
            .map(|sentence| {
                let words_tags = sentence.split_whitespace().collect_vec();

                let mut words = Vec::with_capacity(words_tags.len());
                let mut poses = Vec::with_capacity(words_tags.len());
                let mut labels = Vec::with_capacity(words_tags.len());
                for word_tag in words_tags {
                    let result = word_tag.rsplitn(3, '/');
                    let (label, pos, word) = result.collect_tuple().expect("tag not found");
                    words.push(word);
                    poses.push(pos);
                    labels.push(self.label_to(label));
                }
                let features = self.parse_words_features(&words, &poses);
                (features, labels)
            })
            .collect_vec()
    }

    fn predict(
        &self,
        _: &<Self::RawFeature as GenericItem>::Item,
        _: &<Self::Fragment as GenericItem>::Item,
        predicts: &[usize],
    ) -> Vec<&str> {
        self.to_labels(predicts)
    }

    fn evaluate(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize) {
        self.evaluate_entities(predicts, labels)
    }
}
