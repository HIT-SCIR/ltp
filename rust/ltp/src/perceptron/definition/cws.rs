use crate::perceptron::definition::GenericItem;
use crate::perceptron::{Definition, Sample};
use crate::buf_feature;
use anyhow::Result;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::io::{Write, BufRead, BufReader, Read};

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CWSDefinition {}

impl CWSDefinition {
    pub fn new() -> Self {
        CWSDefinition {}
    }

    /// +--------------+-----------------------------------------------------------------------+
    // | 类别         | 特征                                                                    |
    // +==============+=======================================================================+
    // | char-unigram | ch[-2],ch[-1],ch[0],ch[1],ch[2]                                       |
    // +--------------+-----------------------------------------------------------------------+
    // | char-bigram  | ch[-2]ch[-1],ch[-1]ch[0],ch[0]ch[1],ch[1]ch[2],ch[-2]ch[0],ch[0]ch[2] |
    // +--------------+-----------------------------------------------------------------------+
    // | dulchar      | ch[-1]=ch[0]?                                                        |
    // +--------------+----------------------------------------------------------------------+
    // | dul2char     | ch[-2]=ch[0]?                                                        |
    // +--------------+----------------------------------------------------------------------+
    pub fn parse_char_features(&self, sentence: &str) -> (Vec<usize>, Vec<Vec<String>>) {
        let char_null = '\u{0000}';
        let chars_len = sentence.len();

        let mut index = Vec::with_capacity(chars_len);
        let mut features = Vec::with_capacity(chars_len);

        let mut pre_char = char_null;
        let mut pre2_char = char_null;
        let mut chars = sentence.char_indices().multipeek();
        while let Some((char_idx, cur_char)) = chars.next() {
            if cur_char == ' ' {
                continue;
            }

            let mut feature = Vec::with_capacity(12);

            // ch[0]
            feature.push(format!("2{}", cur_char));

            if pre_char != char_null {
                feature.push(format!("1{}", pre_char)); // ch[-1]
                feature.push(format!("6{}{}", pre_char, cur_char)); // ch[-1]ch[0]

                if pre2_char != char_null {
                    feature.push(format!("0{}", pre2_char)); // ch[-2]
                    feature.push(format!("5{}{}", pre2_char, pre_char)); // ch[-2]ch[-1]
                    feature.push(format!("9{}{}", pre2_char, cur_char)); // ch[-2]ch[0]
                }

                if pre2_char == cur_char {
                    feature.push("c".to_string()); // ch[-2]=ch[0]?
                }
            } else {
                feature.push("BOS".to_string());
            }

            let next_char = if let Some((_, next_char)) = chars.peek() {
                feature.push(format!("3{}", next_char)); // ch[+1]
                feature.push(format!("7{}{}", cur_char, next_char)); // ch[0]ch[+1]
                *next_char
            } else {
                ' '
            };

            if let Some((_, next2_char)) = chars.peek() {
                feature.push(format!("4{}", next2_char)); // ch[+2]
                feature.push(format!("8{}{}", next_char, next2_char)); // ch[+1]ch[+2]
                feature.push(format!("a{}{}", cur_char, next2_char)); // ch[0]ch[+2]
            }

            pre2_char = pre_char;
            pre_char = cur_char;

            index.push(char_idx);
            features.push(feature);
        }
        index.push(chars_len);
        (index, features)
    }

    pub fn parse_char_features_with_buffer<'a>(&self, sentence: &str, buffer: &'a mut Vec<u8>) -> Result<(Vec<usize>, Vec<Vec<&'a str>>)> {
        let char_null = '\u{0000}';
        let chars_len = sentence.len();

        let mut index = Vec::with_capacity(chars_len);
        let mut features = Vec::with_capacity(chars_len);

        let mut pre_char = char_null;
        let mut pre2_char = char_null;
        let mut chars = sentence.char_indices().multipeek();
        while let Some((char_idx, cur_char)) = chars.next() {
            if cur_char == ' ' {
                continue;
            }

            let mut feature = Vec::with_capacity(12);
            // ch[0]
            buf_feature!(buffer, feature, "2{}", cur_char);
            if pre_char != char_null {
                // ch[-1]
                buf_feature!(buffer, feature, "1{}", pre_char);
                // ch[-1]ch[0]
                buf_feature!(buffer, feature, "6{}{}", pre_char, cur_char);

                if pre2_char != char_null {
                    // ch[-2]
                    buf_feature!(buffer, feature, "0{}", pre2_char);
                    // ch[-2]ch[-1]
                    buf_feature!(buffer, feature, "5{}{}", pre2_char, pre_char);
                    // ch[-2]ch[0]
                    buf_feature!(buffer, feature, "9{}{}", pre2_char, cur_char);
                }

                if pre2_char == cur_char {
                    buf_feature!(buffer, feature, "c"); // ch[-2]=ch[0]?
                }
            } else {
                buf_feature!(buffer, feature, "BOS");
            }

            let next_char = if let Some((_, next_char)) = chars.peek() {
                // ch[+1]
                buf_feature!(buffer, feature, "3{}", next_char);
                // ch[0]ch[+1]
                buf_feature!(buffer, feature, "7{}{}", cur_char, next_char);
                *next_char
            } else {
                ' '
            };

            if let Some((_, next2_char)) = chars.peek() {
                // ch[+2]
                buf_feature!(buffer, feature, "4{}", next2_char);
                // ch[+1]ch[+2]
                buf_feature!(buffer, feature, "8{}{}", next_char, next2_char);
                // ch[0]ch[+2]
                buf_feature!(buffer, feature, "a{}{}", cur_char, next2_char);
            }

            pre2_char = pre_char;
            pre_char = cur_char;

            index.push(char_idx);
            features.push(feature);
        }
        index.push(chars_len);

        let mut start = 0usize;
        let mut result = Vec::with_capacity(features.len());
        for feature_end in features {
            let mut feature = Vec::with_capacity(feature_end.len());
            for end in feature_end {
                feature.push(std::str::from_utf8(&buffer[start..end])?);
                start = end;
            }
            result.push(feature);
        }

        Ok((index, result))
    }
}

impl Definition for CWSDefinition {
    type Fragment = dyn for<'any> GenericItem<'any, Item=Vec<usize>>;
    type Prediction = dyn for<'any> GenericItem<'any, Item=Vec<&'any str>>;
    type RawFeature = dyn for<'any> GenericItem<'any, Item=&'any str>;

    fn use_viterbi(&self) -> bool {
        true
    }

    fn labels(&self) -> Vec<String> {
        vec![
            "S".to_string(),
            "B".to_string(),
            "M".to_string(),
            "E".to_string(),
        ]
    }

    fn label_num(&self) -> usize {
        4
    }

    fn label_to(&self, label: &str) -> usize {
        match label {
            "S" => 0,
            "B" => 1,
            "M" => 2,
            "E" => 3,
            _ => panic!("unknown label"),
        }
    }

    fn to_label(&self, index: usize) -> &str {
        match index {
            0 => "S",
            1 => "B",
            2 => "M",
            3 => "E",
            _ => panic!("unknown label index"),
        }
    }

    fn parse_features(&self, sentence: &&str) -> (Vec<usize>, Vec<Vec<String>>) {
        let (index, features) = self.parse_char_features(sentence);
        (index, features)
    }

    fn parse_features_with_buffer<'a>(&self, sentence: &&str, buf: &'a mut Vec<u8>) -> Result<(Vec<usize>, Vec<Vec<&'a str>>)> {
        let (index, features) = self.parse_char_features_with_buffer(sentence, buf)?;
        Ok((index, features))
    }

    #[cfg(feature = "parallel")]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Vec<Sample> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();
        let mut result = Vec::with_capacity(lines.len());

        lines
            .par_iter()
            .map(|sentence| {
                let (_, features) = self.parse_char_features(sentence.as_str());
                let mut labels = Vec::with_capacity(features.len());
                // 构建标签序列

                let mut last_char = ' ';
                let mut chars = sentence.chars().peekable();
                while let Some(cur_char) = chars.next() {
                    if cur_char == ' ' {
                        last_char = cur_char;
                        continue;
                    }
                    if let Some(next_char) = chars.peek() {
                        match (last_char, next_char) {
                            (' ', ' ') => labels.push(self.label_to("S")),
                            (' ', _nc) => labels.push(self.label_to("B")),
                            (_lc, ' ') => labels.push(self.label_to("E")),
                            (_lc, _nc) => labels.push(self.label_to("M")),
                        }
                    } else if last_char == ' ' {
                        labels.push(self.label_to("S"));
                    } else {
                        labels.push(self.label_to("E"));
                    }
                    last_char = cur_char;
                }
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
                let (_, features) = self.parse_char_features(sentence.as_str());
                let mut labels = Vec::with_capacity(features.len());
                // 构建标签序列

                let mut last_char = ' ';
                let mut chars = sentence.chars().peekable();
                while let Some(cur_char) = chars.next() {
                    if cur_char == ' ' {
                        last_char = cur_char;
                        continue;
                    }
                    if let Some(next_char) = chars.peek() {
                        match (last_char, next_char) {
                            (' ', ' ') => labels.push(self.label_to("S")),
                            (' ', _nc) => labels.push(self.label_to("B")),
                            (_lc, ' ') => labels.push(self.label_to("E")),
                            (_lc, _nc) => labels.push(self.label_to("M")),
                        }
                    } else if last_char == ' ' {
                        labels.push(self.label_to("S"));
                    } else {
                        labels.push(self.label_to("E"));
                    }
                    last_char = cur_char;
                }
                (features, labels)
            })
            .collect()
    }

    fn predict(&self, _: &&str, _: &Vec<usize>, predicts: &[usize]) -> Vec<&str> {
        self.to_labels(predicts)
    }

    fn evaluate(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize) {
        self.evaluate_entities(predicts, labels)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;
    use super::CWSDefinition as Define;
    use anyhow::Result;

    #[test]
    fn test_vec_buffer() -> Result<()> {
        let mut buffer = Vec::new();

        let sentence = "桂林警备区从一九九○年以来，先后修建水电站十五座，整修水渠六千七百四十公里，兴修水利一千五百六十五处，修建机耕路一百二十六公里，修建人畜饮水工程二百六十五处，解决饮水人口六点五万人，使八万多壮、瑶、苗、侗、回等民族的群众脱了贫，占桂林地、市脱贫人口总数的百分之三十七点六。";
        let define = Define::default();
        let (_, no_buffer) = define.parse_char_features(sentence);
        let (_, with_buffer) = define.parse_char_features_with_buffer(sentence, &mut buffer)?;

        for (a, b) in zip(no_buffer, with_buffer) {
            for (c, d) in zip(a, b) {
                assert_eq!(c, d);
            }
        }

        println!("{}/{}/{}", sentence.len(), buffer.len(), buffer.len() / sentence.len());

        Ok(())
    }
}
