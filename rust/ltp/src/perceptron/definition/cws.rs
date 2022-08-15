use crate::get_entities;
use crate::perceptron::definition::GenericItem;
use crate::perceptron::{Definition, Sample};
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};

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
    pub fn parse_char_features(&self, chars: &[char]) -> Vec<Vec<String>> {
        let char_null = '\u{0000}';
        let chars_len = chars.len();
        let mut features = Vec::with_capacity(chars_len);
        for (idx, cur_char) in chars.iter().enumerate() {
            // 剩余字符数
            let last = chars_len - idx - 1;
            let pre2_char = if idx > 1 { chars[idx - 2] } else { char_null };
            let pre_char = if idx > 0 { chars[idx - 1] } else { char_null };
            let next_char = if last > 0 { chars[idx + 1] } else { char_null };
            let next2_char = if last > 1 { chars[idx + 2] } else { char_null };

            // todo: 优化容量设置
            let mut feature = Vec::with_capacity(12);
            feature.push(format!("2{}", cur_char));
            if idx > 0 {
                feature.push(format!("1{}", pre_char)); // ch[-1]
                feature.push(format!("6{}{}", pre_char, cur_char)); // ch[-1]ch[0]

                if pre_char == *cur_char {
                    feature.push("b".to_string()); // ch[-1]=ch[0]?
                }

                if idx > 1 {
                    feature.push(format!("0{}", pre2_char)); // ch[-2]
                    feature.push(format!("5{}{}", pre2_char, pre_char)); // ch[-2]ch[-1]
                    feature.push(format!("9{}{}", pre2_char, cur_char)); // ch[-2]ch[0]

                    if pre2_char == *cur_char {
                        feature.push("c".to_string()); // ch[-2]=ch[0]?
                    }
                }
            } else {
                feature.push("BOS".to_string());
            }

            if last > 0 {
                feature.push(format!("3{}", next_char)); // ch[+1]
                feature.push(format!("7{}{}", cur_char, next_char)); // ch[0]ch[+1]

                if last > 1 {
                    feature.push(format!("4{}", next2_char)); // ch[+2]
                    feature.push(format!("8{}{}", next_char, next2_char)); // ch[+1]ch[+2]
                    feature.push(format!("a{}{}", cur_char, next2_char)); // ch[0]ch[+2]
                }
            }

            features.push(feature);
        }
        features
    }
}

impl Definition for CWSDefinition {
    type Fragment = dyn for<'any> GenericItem<'any, Item = Vec<char>>;
    type Prediction = dyn for<'any> GenericItem<'any, Item = Vec<String>>;
    type RawFeature = dyn for<'any> GenericItem<'any, Item = &'any str>;

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

    fn parse_features(&self, line: &str) -> (Vec<char>, Vec<Vec<String>>) {
        let chars = line.chars().collect::<Vec<_>>();
        let words = chars
            .split(|&char| char.is_whitespace())
            .filter(|&char| !char.is_empty())
            .collect::<Vec<_>>();
        let chars = words.concat();
        let features = self.parse_char_features(&chars);

        (chars, features)
    }

    #[cfg(feature = "parallel")]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Vec<Sample> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();
        let mut result = Vec::with_capacity(lines.len());

        lines
            .par_iter()
            .map(|sentence| {
                let chars = sentence.chars().collect::<Vec<_>>();
                let words = chars
                    .split(|&char| char.is_whitespace())
                    .filter(|&char| !char.is_empty())
                    .collect::<Vec<_>>();
                let chars = words.concat();
                let features = self.parse_char_features(&chars);

                // 构建标签序列
                let mut labels = Vec::with_capacity(chars.len());
                for word in words {
                    match word.len() {
                        1 => {
                            labels.push(self.label_to("S"));
                        }
                        2 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("E"));
                        }
                        3 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        4 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        5 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        length => {
                            labels.push(self.label_to("B"));
                            for _ in 1..length - 1 {
                                labels.push(self.label_to("M"));
                            }
                            labels.push(self.label_to("E"));
                        }
                    }
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
                let chars = sentence.chars().collect::<Vec<_>>();
                let words = chars
                    .split(|&char| char.is_whitespace())
                    .filter(|&char| !char.is_empty())
                    .collect::<Vec<_>>();
                let chars = words.concat();
                let features = self.parse_char_features(&chars);

                // 构建标签序列
                let mut labels = Vec::with_capacity(chars.len());
                for word in words {
                    match word.len() {
                        1 => {
                            labels.push(self.label_to("S"));
                        }
                        2 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("E"));
                        }
                        3 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        4 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        5 => {
                            labels.push(self.label_to("B"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("M"));
                            labels.push(self.label_to("E"));
                        }
                        length => {
                            labels.push(self.label_to("B"));
                            for _ in 1..length - 1 {
                                labels.push(self.label_to("M"));
                            }
                            labels.push(self.label_to("E"));
                        }
                    }
                }
                (features, labels)
            })
            .collect()
    }

    fn predict(&self, fragments: &Vec<char>, predicts: &[usize]) -> Vec<String> {
        let predicts = self.to_labels(predicts);
        let predicts = get_entities(&predicts);

        predicts
            .into_iter()
            .map(|(_, start, end)| fragments.iter().take(end + 1).skip(start).collect())
            .collect::<Vec<_>>()
    }

    fn evaluate(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize) {
        self.evaluate_entities(predicts, labels)
    }
}
