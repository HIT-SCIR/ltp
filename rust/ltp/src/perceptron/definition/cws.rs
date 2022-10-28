use crate::buf_feature;
use crate::perceptron::definition::GenericItem;
use crate::perceptron::{Definition, Sample};
use anyhow::Result;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};

/// Character type.
#[cfg(any(
    feature = "char-type",
    feature = "cross-char",
    feature = "near-char-type"
))]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum CharacterType {
    /// Digit character. (e.g. 0, 1, 2, ...)
    Digit = 1,

    /// Roman character. (e.g. A, B, C, ...)
    Roman = 2,

    /// Japanese Hiragana character. (e.g. あ, い, う, ...)
    Hiragana = 3,

    /// Japanese Katakana character. (e.g. ア, イ, ウ, ...)
    Katakana = 4,

    /// Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
    Kanji = 5,

    /// Other character.
    Other = 6,
}

#[cfg(any(
    feature = "char-type",
    feature = "cross-char",
    feature = "near-char-type"
))]
impl CharacterType {
    pub fn get_type(c: char) -> Self {
        match u32::from(c) {
            0x30..=0x39 | 0xFF10..=0xFF19 => Self::Digit,
            0x41..=0x5A | 0x61..=0x7A | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A => Self::Roman,
            0x3040..=0x3096 => Self::Hiragana,
            0x30A0..=0x30FA | 0x30FC..=0x30FF | 0xFF66..=0xFF9F => Self::Katakana,
            0x3400..=0x4DBF      // CJK Unified Ideographs Extension A
            | 0x4E00..=0x9FFF    // CJK Unified Ideographs
            | 0xF900..=0xFAFF    // CJK Compatibility Ideographs
            | 0x20000..=0x2A6DF  // CJK Unified Ideographs Extension B
            | 0x2A700..=0x2B73F  // CJK Unified Ideographs Extension C
            | 0x2B740..=0x2B81F  // CJK Unified Ideographs Extension D
            | 0x2B820..=0x2CEAF  // CJK Unified Ideographs Extension E
            | 0x2F800..=0x2FA1F  // CJK Compatibility Ideographs Supplement
            => Self::Kanji,
            _ => Self::Other,
        }
    }
}

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
    pub fn parse_char_features_with_buffer<'a>(
        &self,
        sentence: &str,
        buffer: &'a mut Vec<u8>,
    ) -> Result<(Vec<usize>, Vec<Vec<usize>>)> {
        let char_null = '\u{0000}';
        let chars_len = sentence.len();

        let mut index = Vec::with_capacity(chars_len);
        let mut features = Vec::with_capacity(chars_len);

        let mut pre_char = char_null;
        let mut pre2_char = char_null;
        let mut chars = sentence
            .char_indices()
            .filter(|(_, ch)| !ch.is_whitespace())
            .multipeek();
        while let Some((char_idx, cur_char)) = chars.next() {
            let mut feature = Vec::with_capacity(13);
            // ch[0]
            buf_feature!(buffer, feature, "2{}", cur_char);
            // TYPE(ch[0])
            #[cfg(feature = "char-type")]
            buf_feature!(
                buffer,
                feature,
                "b{}",
                CharacterType::get_type(cur_char) as u8
            );
            if pre_char != char_null {
                // ch[-1]
                buf_feature!(buffer, feature, "1{}", pre_char);
                // ch[-1]ch[0]
                buf_feature!(buffer, feature, "6{}{}", pre_char, cur_char);
                // TYPE(ch[-1])
                #[cfg(feature = "char-type")]
                buf_feature!(
                    buffer,
                    feature,
                    "c{}",
                    CharacterType::get_type(pre_char) as u8
                );

                // TYPE(ch[-1]) TYPE(ch[0])
                #[cfg(feature = "near-char-type")]
                buf_feature!(
                    buffer,
                    feature,
                    "d{}{}",
                    CharacterType::get_type(pre_char) as u8,
                    CharacterType::get_type(cur_char) as u8
                );

                if pre2_char != char_null {
                    // ch[-2]
                    buf_feature!(buffer, feature, "0{}", pre2_char);
                    // ch[-2]ch[-1]
                    buf_feature!(buffer, feature, "5{}{}", pre2_char, pre_char);
                    // ch[-2]ch[0]
                    #[cfg(feature = "cross-char")]
                    buf_feature!(buffer, feature, "9{}{}", pre2_char, cur_char);
                }

                if pre2_char == cur_char {
                    buf_feature!(buffer, feature, "c"); // ch[-2]=ch[0]?
                }
            }

            let next_char = if let Some((_, next_char)) = chars.peek() {
                // ch[+1]
                buf_feature!(buffer, feature, "3{}", next_char);
                // ch[0]ch[+1]
                buf_feature!(buffer, feature, "7{}{}", cur_char, next_char);
                // TYPE(ch[1])
                #[cfg(feature = "char-type")]
                buf_feature!(
                    buffer,
                    feature,
                    "d{}",
                    CharacterType::get_type(*next_char) as u8
                );
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
                #[cfg(feature = "cross-char")]
                buf_feature!(buffer, feature, "a{}{}", cur_char, next2_char);
            }

            pre2_char = pre_char;
            pre_char = cur_char;

            index.push(char_idx);
            features.push(feature);
        }
        index.push(chars_len);
        Ok((index, features))
    }

    pub fn parse_char_features(&self, sentence: &str) -> Result<(Vec<usize>, Vec<Vec<String>>)> {
        let mut buffer = Vec::with_capacity(sentence.len() * 20);
        let (index, features) = self.parse_char_features_with_buffer(sentence, &mut buffer)?;

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

        Ok((index, result))
    }

    pub fn parse_char_features_with_buffer_str<'a>(
        &self,
        sentence: &str,
        buffer: &'a mut Vec<u8>,
    ) -> Result<(Vec<usize>, Vec<Vec<&'a str>>)> {
        let (index, features) = self.parse_char_features_with_buffer(sentence, buffer)?;

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

        Ok((index, result))
    }
}

impl Definition for CWSDefinition {
    type Fragment = dyn for<'any> GenericItem<'any, Item = Vec<usize>>;
    type Prediction = dyn for<'any> GenericItem<'any, Item = Vec<&'any str>>;
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

    fn parse_features(&self, sentence: &&str) -> Result<(Vec<usize>, Vec<Vec<String>>)> {
        let (index, features) = self.parse_char_features(sentence)?;
        Ok((index, features))
    }

    fn parse_features_with_buffer<'a>(
        &self,
        sentence: &&str,
        buf: &'a mut Vec<u8>,
    ) -> Result<(Vec<usize>, Vec<Vec<&'a str>>)> {
        let (index, features) = self.parse_char_features_with_buffer_str(sentence, buf)?;
        Ok((index, features))
    }

    #[cfg(feature = "parallel")]
    fn parse_gold_features<R: Read>(&self, reader: R) -> Result<Vec<Sample>> {
        let lines = BufReader::new(reader).lines();
        let lines = lines.flatten().filter(|s| !s.is_empty()).collect_vec();

        lines
            .par_iter()
            .map(|sentence| {
                self.parse_char_features(sentence).map(|(_, features)| {
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
                self.parse_char_features(sentence).map(|(_, features)| {
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
    use super::CWSDefinition as Define;
    use anyhow::Result;
    use std::iter::zip;

    #[test]
    fn test_vec_buffer() -> Result<()> {
        let mut buffer = Vec::new();

        let sentence = "桂林警备区从一九九○年以来，先后修建水电站十五座，整修水渠六千七百四十公里，兴修水利一千五百六十五处，修建机耕路一百二十六公里，修建人畜饮水工程二百六十五处，解决饮水人口六点五万人，使八万多壮、瑶、苗、侗、回等民族的群众脱了贫，占桂林地、市脱贫人口总数的百分之三十七点六。";
        let define = Define::default();
        let (_, no_buffer) = define.parse_char_features(sentence)?;
        let (_, with_buffer) = define.parse_char_features_with_buffer_str(sentence, &mut buffer)?;

        for (a, b) in zip(no_buffer, with_buffer) {
            for (c, d) in zip(a, b) {
                assert_eq!(c, d);
            }
        }

        println!(
            "{}/{}/{}",
            sentence.len(),
            buffer.len(),
            buffer.len() / sentence.len()
        );

        Ok(())
    }

    #[test]
    fn test_features() -> Result<()> {
        let define = Define::default();
        let (_, num_han) = define.parse_char_features("1汉")?;
        let (_, roman_han) = define.parse_char_features("a汉")?;
        let (_, num_roman) = define.parse_char_features("1a")?;

        println!("数字+汉字: {:?}", num_han);
        println!("字母+汉字: {:?}", roman_han);
        println!("数字+字母: {:?}", num_roman);

        Ok(())
    }
}
