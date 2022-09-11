use cedarwood::Cedar;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Record {
    freq: usize,
}

impl Record {
    #[inline(always)]
    fn new(freq: usize) -> Self {
        Self { freq }
    }
}

#[derive(Clone, Debug)]
pub struct Hook {
    records: Vec<Record>,
    cedar: Cedar,
    total: usize,
    longest_word_len: usize,
}

impl Default for Hook {
    fn default() -> Self {
        Self::new()
    }
}


impl Hook {
    pub fn new() -> Hook {
        Hook {
            records: Vec::new(),
            cedar: Cedar::new(),
            total: 0,
            longest_word_len: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn add_word(&mut self, word: &str, freq: Option<usize>) -> usize {
        let freq = freq.unwrap_or(1);

        match self.cedar.exact_match_search(word) {
            Some((word_id, _, _)) => {
                let old_freq = self.records[word_id as usize].freq;
                self.records[word_id as usize].freq = freq;

                self.total += freq;
                self.total -= old_freq;
            }
            None => {
                self.records.push(Record::new(freq));
                let word_id = (self.records.len() - 1) as i32;

                self.cedar.update(word, word_id);
                self.total += freq;
            }
        };

        let curr_word_len = word.chars().count();
        if self.longest_word_len < curr_word_len {
            self.longest_word_len = curr_word_len;
        }

        freq
    }

    fn dag(&self, sentence: &str, words: &[&str], dag: &mut Dag) {
        let mut byte_start_bias = 0;
        for &word in words {
            let word_len = word.len();
            let is_first = true;
            let mut char_indices = word.char_indices().peekable();
            while let Some((byte_start, _)) = char_indices.next() {
                dag.start(byte_start + byte_start_bias);
                let haystack = &sentence[byte_start + byte_start_bias..];

                // Char
                let cur_char_len = char_indices.peek().map(|(next_start, _)| next_start - byte_start);
                // 外部分词结果
                let mut nch_flag = cur_char_len.is_none();
                let mut per_flag = !is_first;
                for (_, end_index) in self.cedar.common_prefix_iter(haystack) {
                    let white_space_len = haystack[end_index + 1..].chars().take_while(|ch| ch.is_whitespace()).count();
                    if is_first && end_index + white_space_len + 1 == word_len {
                        per_flag = true;
                    }
                    if let Some(char_len) = cur_char_len {
                        if end_index + white_space_len + 1 == char_len {
                            nch_flag = true;
                        }
                    }
                    dag.insert(byte_start_bias + byte_start + end_index + white_space_len + 1);
                }
                if !nch_flag {
                    dag.insert(byte_start_bias + byte_start + cur_char_len.unwrap());
                    if byte_start + cur_char_len.unwrap() == word_len {
                        per_flag = true;
                    }
                }
                if is_first && !per_flag {
                    dag.insert(byte_start_bias + word_len);
                }
                dag.commit();
            }
            byte_start_bias += word_len;
        }
    }

    #[allow(clippy::ptr_arg)]
    fn calc(&self, sentence: &str, dag: &Dag, route: &mut Vec<(f64, usize)>) {
        let str_len = sentence.len();

        if str_len + 1 > route.len() {
            route.resize(str_len + 1, (0.0, 0));
        }

        let logtotal = (self.total as f64).ln();
        let mut prev_byte_start = str_len;
        let curr = sentence.char_indices().map(|x| x.0).rev();
        for byte_start in curr {
            let pair = dag
                .iter_edges(byte_start)
                .map(|byte_end| {
                    let wfrag = if byte_end == str_len {
                        &sentence[byte_start..]
                    } else {
                        &sentence[byte_start..byte_end]
                    };

                    let freq = if let Some((word_id, _, _)) = self.cedar.exact_match_search(wfrag) {
                        self.records[word_id as usize].freq
                    } else {
                        1
                    };

                    ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

            if let Some(p) = pair {
                route[byte_start] = p;
            } else {
                let byte_end = prev_byte_start;
                let freq = 1;
                route[byte_start] = ((freq as f64).ln() - logtotal + route[byte_end].0, byte_end);
            }

            prev_byte_start = byte_start;
        }
    }

    pub fn hook<'a>(&self, sentence: &'a str, cut_words: &[&str]) -> Vec<&'a str> {
        let mut hook_words = Vec::with_capacity(cut_words.len());
        let mut route = Vec::with_capacity(cut_words.len());
        let mut dag = Dag::with_size_hint(cut_words.len());

        self.inner_hook(sentence, cut_words, &mut hook_words, &mut route, &mut dag);
        hook_words
    }

    fn inner_hook<'a>(
        &self,
        sentence: &'a str,
        cut_words: &[&str],
        words: &mut Vec<&'a str>,
        route: &mut Vec<(f64, usize)>,
        dag: &mut Dag,
    ) {
        self.dag(sentence, cut_words, dag);
        self.calc(sentence, dag, route);
        let mut x = 0;
        let mut left: Option<usize> = None;

        while x < sentence.len() {
            let y = route[x].1;
            let l_str = if y < sentence.len() {
                &sentence[x..y]
            } else {
                &sentence[x..]
            };

            if l_str.chars().count() == 1 && l_str.chars().all(|ch| ch.is_ascii_alphanumeric()) {
                if left.is_none() {
                    left = Some(x);
                }
            } else {
                if let Some(byte_start) = left {
                    let word = &sentence[byte_start..x];
                    words.push(word);
                    left = None;
                }

                let word = if y < sentence.len() {
                    &sentence[x..y]
                } else {
                    &sentence[x..]
                };

                words.push(word);
            }
            x = y;
        }

        if let Some(byte_start) = left {
            let word = &sentence[byte_start..];
            words.push(word);
        }

        dag.clear();
        route.clear();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dag {
    array: Vec<usize>,
    start_pos: HashMap<usize, usize>,
    size_hint_for_iterator: usize,
    curr_insertion_len: usize,
}

pub struct EdgeIter<'a> {
    dag: &'a Dag,
    cursor: usize,
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = usize;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.dag.size_hint_for_iterator))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.dag.array[self.cursor] == 0 {
            self.cursor += 1;
            None
        } else {
            let v = self.dag.array[self.cursor] - 1;
            self.cursor += 1;
            Some(v)
        }
    }
}

impl Dag {
    pub(crate) fn with_size_hint(hint: usize) -> Self {
        Dag {
            array: Vec::with_capacity(hint * 5),
            start_pos: HashMap::default(),
            size_hint_for_iterator: 0,
            curr_insertion_len: 0,
        }
    }

    #[inline]
    pub(crate) fn start(&mut self, from: usize) {
        let idx = self.array.len();
        self.curr_insertion_len = 0;
        self.start_pos.insert(from, idx);
    }

    #[inline]
    pub(crate) fn insert(&mut self, to: usize) {
        self.curr_insertion_len += 1;
        self.array.push(to + 1);
    }

    #[inline]
    pub(crate) fn commit(&mut self) {
        self.size_hint_for_iterator =
            std::cmp::max(self.curr_insertion_len, self.size_hint_for_iterator);
        self.array.push(0);
    }

    #[inline]
    pub(crate) fn iter_edges(&self, from: usize) -> EdgeIter {
        let cursor = self.start_pos.get(&from).unwrap().to_owned();

        EdgeIter { dag: self, cursor }
    }

    pub(crate) fn clear(&mut self) {
        self.array.clear();
        self.start_pos.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook() {
        let sentence = "他叫汤姆去拿外衣。";
        let cut_words = ["他", "叫", "汤姆", "去", "拿", "外衣", "。"];
        let mut hook = Hook::new();

        let mut words = Vec::with_capacity(5);
        let mut route = Vec::with_capacity(5);

        let mut dag = Dag::with_size_hint(5);
        hook.inner_hook(sentence, &cut_words, &mut words, &mut route, &mut dag);

        assert_eq!(words, cut_words);

        hook.add_word("姆去拿", Some(2));
        words.clear();
        route.clear();
        dag.clear();

        hook.inner_hook(sentence, &cut_words, &mut words, &mut route, &mut dag);
        println!("{:?}", words);
        assert_eq!(words, ["他", "叫", "汤", "姆去拿", "外衣", "。"]);
    }

    #[test]
    fn test_sep() {
        let sentence = "通讯系统[SEP]";
        let cut_words = ["通讯", "系统[SEP]"];
        let hook = Hook::new();

        let mut words = Vec::with_capacity(5);
        let mut route = Vec::with_capacity(5);

        let mut dag = Dag::with_size_hint(5);
        hook.inner_hook(sentence, &cut_words, &mut words, &mut route, &mut dag);
    }

    #[test]
    fn test_space() {
        let sentence = "[ENT] Info";
        let cut_words = ["[", "ENT", "] Info"];
        let mut hook = Hook::new();
        hook.add_word("[ENT]", Some(2));

        let mut words = Vec::with_capacity(5);
        let mut route = Vec::with_capacity(5);

        let mut dag = Dag::with_size_hint(5);
        hook.inner_hook(sentence, &cut_words, &mut words, &mut route, &mut dag);
        println!("{:?}", words);
    }

    #[test]
    fn test_dag() {
        let mut dag = Dag::with_size_hint(5);
        let mut ans: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for i in 0..=3 {
            dag.start(i);
            for j in (i + 1)..=4 {
                ans[i].push(j);
                dag.insert(j);
            }

            dag.commit()
        }

        assert_eq!(dag.size_hint_for_iterator, 4);

        for i in 0..=3 {
            let edges: Vec<usize> = dag.iter_edges(i).collect();
            assert_eq!(ans[i], edges);
        }
    }
}
