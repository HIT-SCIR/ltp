use crate::perceptron::definition::GenericItem;
use crate::perceptron::{Definition, Sample};
use crate::buf_feature;
use anyhow::Result;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};


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
    pub fn parse_words_features_with_buffer<'a>(&self, words: &[&str], poses: &[&str], buffer: &'a mut Vec<u8>) -> Result<Vec<Vec<usize>>> {
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
            buf_feature!(buffer, feature, "2{}", words[idx]);

            // p[0]
            buf_feature!(buffer, feature, "d{}", poses[idx]);

            if idx > 0 {
                buf_feature!(buffer, feature, "1{}", pre_word);
                // w[-1]
                buf_feature!(buffer, feature, "6{}{}", pre_word, cur_word);
                // w[-1]w[0]
                buf_feature!(buffer, feature, "c{}", poses[idx - 1]);
                // p[-1]
                buf_feature!(buffer, feature, "g{}{}", poses[idx - 1], poses[idx]); // p[-1]p[0]
                if idx > 1 {
                    buf_feature!(buffer, feature, "0{}", pre2_word);
                    // w[-2]
                    buf_feature!(buffer, feature, "5{}{}", pre2_word, pre_word);
                    // w[-2]w[-1]
                    buf_feature!(buffer, feature, "9{}{}", pre2_word, cur_word); // w[-2]w[0]

                    buf_feature!(buffer, feature, "b{}", poses[idx - 2]); // p[-2]
                }
            }

            if last > 0 {
                buf_feature!(buffer, feature, "3{}", next_word);
                // w[+1]
                buf_feature!(buffer, feature, "7{}{}", cur_word, next_word);
                // w[0]w[+1]
                buf_feature!(buffer, feature, "e{}", poses[idx + 1]);
                // p[+1]
                buf_feature!(buffer, feature, "h{}{}", poses[idx], poses[idx + 1]); // p[0]p[+1]
                if last > 1 {
                    buf_feature!(buffer, feature, "4{}", next2_word);
                    // w[+2]
                    buf_feature!(buffer, feature, "8{}{}", next_word, next2_word);
                    // w[+1]w[+2]
                    buf_feature!(buffer, feature, "a{}{}", cur_word, next2_word);
                    // w[0]w[+2]
                    buf_feature!(buffer, feature, "f{}", poses[idx + 2]); // p[+2]
                }
            }

            features.push(feature);
        }

        Ok(features)
    }

    pub fn parse_words_features(&self, words: &[&str], poses: &[&str]) -> Result<Vec<Vec<String>>> {
        let mut buffer = Vec::with_capacity(words.len() * 150);
        let features = self.parse_words_features_with_buffer(words, poses, &mut buffer)?;

        let mut start = 0usize;
        let mut result = Vec::with_capacity(features.len());
        for feature_end in features {
            let mut feature = Vec::with_capacity(feature_end.len());
            for end in feature_end {
                // Safety : all write are valid utf8
                feature.push(String::from_utf8_lossy(&buffer[start..end]).to_string());
                start = end;
            }
            result.push(feature);
        }
        Ok(result)
    }

    pub fn parse_words_features_with_buffer_str<'a>(&self, words: &[&str], poses: &[&str], buffer: &'a mut Vec<u8>) -> Result<Vec<Vec<&'a str>>> {
        let features = self.parse_words_features_with_buffer(words, poses, buffer)?;

        let mut start = 0usize;
        let mut result = Vec::with_capacity(features.len());
        for feature_end in features {
            let mut feature = Vec::with_capacity(feature_end.len());
            for end in feature_end {
                // Safety : all write are valid utf8
                feature.push(unsafe { std::str::from_utf8_unchecked(&buffer[start..end]) });
                start = end;
            }
            result.push(feature);
        }
        Ok(result)
    }
}

impl Definition for NERDefinition {
    type Fragment = dyn for<'any> GenericItem<'any, Item=()>;
    type Prediction = dyn for<'any> GenericItem<'any, Item=Vec<&'any str>>;
    type RawFeature =
    dyn for<'any> GenericItem<'any, Item=(&'any [&'any str], &'any [&'any str])>;

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
    ) -> Result<((), Vec<Vec<String>>)> {
        let (words, features) = line;
        let features = self.parse_words_features(words, features)?;

        Ok(((), features))
    }

    fn parse_features_with_buffer<'a>(
        &self,
        line: &<Self::RawFeature as GenericItem>::Item,
        buf: &'a mut Vec<u8>,
    ) -> Result<((), Vec<Vec<&'a str>>)> {
        let (words, features) = line;
        let features = self.parse_words_features_with_buffer_str(words, features, buf)?;
        Ok(((), features))
    }

    #[cfg(feature = "parallel")]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Result<Vec<Sample>> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();

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
                self.parse_words_features(&words, &poses).map(|features| { (features, labels) })
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Result<Vec<Sample>> {
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
                self.parse_words_features(&words, &poses).map(|features| { (features, labels) })
            })
            .collect()
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


#[cfg(test)]
mod tests {
    use std::iter::zip;
    use super::NERDefinition as Define;
    use anyhow::Result;

    #[test]
    fn test_vec_buffer() -> Result<()> {
        let mut buffer = Vec::new();
        let sentence = vec!["桂林", "警备区", "从", "一九九○年", "以来", "，", "先后", "修建", "水电站", "十五", "座", "，", "整修", "水渠", "六千七百四十", "公里", "，", "兴修", "水利", "一千五百六十五", "处", "，", "修建", "机耕路", "一百二十六", "公里", "，", "修建", "人", "畜", "饮水", "工程", "二百六十五", "处", "，", "解决", "饮水", "人口", "六点五万", "人", "，", "使", "八万", "多", "壮", "、", "瑶", "、", "苗", "、", "侗", "、", "回", "等", "民族", "的", "群众", "脱", "了", "贫", "，", "占", "桂林", "地", "、", "市", "脱贫", "人口", "总数", "的", "百分之三十七点六", "。"];
        let pos = vec!["ns", "n", "p", "nt", "nd", "wp", "d", "v", "n", "m", "q", "wp", "v", "n", "m", "q", "wp", "v", "n", "m", "q", "wp", "v", "n", "m", "q", "wp", "v", "n", "n", "n", "n", "m", "q", "wp", "v", "n", "n", "m", "n", "wp", "v", "m", "m", "j", "wp", "j", "wp", "j", "wp", "j", "wp", "j", "u", "n", "u", "n", "v", "u", "a", "wp", "v", "ns", "n", "wp", "n", "v", "n", "n", "u", "m", "wp"];
        let define = Define::default();
        let no_buffer = define.parse_words_features(&sentence, &pos)?;
        let with_buffer = define.parse_words_features_with_buffer_str(&sentence, &pos, &mut buffer)?;

        for (a, b) in zip(no_buffer, with_buffer) {
            for (c, d) in zip(a, b) {
                assert_eq!(c, d);
            }
        }

        println!("{}/{}/{}", sentence.len(), buffer.len(), buffer.len() / sentence.len());

        Ok(())
    }
}
