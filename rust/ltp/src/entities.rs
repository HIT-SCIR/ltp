use crate::perceptron::TraitParameterStorageUtils;
use std::borrow::Cow;
use std::ops::Index;

fn start_of_chunk(prev_tag: &str, tag: &str, prev_type: Option<&str>, type_: Option<&str>) -> bool {
    if tag == "B" {
        return true;
    }
    if tag == "S" {
        return true;
    }

    if prev_tag == "E" && tag == "E" {
        return true;
    }
    if prev_tag == "E" && tag == "I" {
        return true;
    }
    if prev_tag == "E" && tag == "M" {
        return true;
    }
    if prev_tag == "S" && tag == "E" {
        return true;
    }
    if prev_tag == "S" && tag == "I" {
        return true;
    }
    if prev_tag == "S" && tag == "M" {
        return true;
    }
    if prev_tag == "O" && tag == "E" {
        return true;
    }
    if prev_tag == "O" && tag == "I" {
        return true;
    }
    if prev_tag == "O" && tag == "M" {
        return true;
    }

    if tag != "O" && prev_type != type_ {
        return true;
    }
    false
}

fn end_of_chunk(prev_tag: &str, tag: &str, prev_type: Option<&str>, type_: Option<&str>) -> bool {
    if prev_tag == "E" {
        return true;
    }
    if prev_tag == "S" {
        return true;
    }

    if prev_tag == "B" && tag == "B" {
        return true;
    }
    if prev_tag == "B" && tag == "S" {
        return true;
    }
    if prev_tag == "B" && tag == "O" {
        return true;
    }
    if prev_tag == "I" && tag == "B" {
        return true;
    }
    if prev_tag == "M" && tag == "B" {
        return true;
    }
    if prev_tag == "I" && tag == "S" {
        return true;
    }
    if prev_tag == "M" && tag == "S" {
        return true;
    }
    if prev_tag == "I" && tag == "O" {
        return true;
    }
    if prev_tag == "M" && tag == "O" {
        return true;
    }

    if prev_tag != "O" && prev_type != type_ {
        return true;
    }
    false
}

pub trait GetEntities {
    fn get_entities(&self) -> Vec<(&str, usize, usize)>;
}

pub fn get_entities<'a, T>(seq: &'a T) -> Vec<(&'a str, usize, usize)>
where
    T: 'a + GetEntities,
{
    seq.get_entities()
}

macro_rules! impl_get_entities {
    ($t:ident) => {{
        let mut prev_tag = "O";
        let mut prev_type = None;
        let mut begin_offset: usize = 0;
        let mut chunks: Vec<(&str, usize, usize)> = Vec::new();
        let length = $t.len();

        for i in 0..length {
            let chunk = &$t[i];
            let cut = chunk.find('-');
            let (tag, type_) = match cut {
                None => match &chunk[0..] {
                    "O" => (&chunk[0..], None),
                    _ => (&chunk[0..], Some("_"))
                },
                Some(cut) => (&chunk[..cut], Some(&chunk[cut + 1..])),
            };
            if end_of_chunk(prev_tag, tag, prev_type, type_) {
                if let Some(prev_type) = prev_type {
                    chunks.push((prev_type, begin_offset, i - 1));
                }
            }
            if start_of_chunk(prev_tag, tag, prev_type, type_) {
                begin_offset = i;
            }
            prev_tag = tag;
            prev_type = type_;

            if i == length - 1 {
                if let Some(type_) = type_ {
                    chunks.push((type_, begin_offset, i));
                }
            }
        }
        chunks
    }};
    ($t:ty) => {
        impl GetEntities for $t {
            fn get_entities(&self) -> Vec<(&str, usize, usize)> {
                impl_get_entities!(self)
            }
        }
    };
    ($t:ty, $($rest:ty),+) => {
        impl_get_entities!($t);
        impl_get_entities!($($rest),+);
    };
}

pub fn drop_get_entities<'a, T>(tags: T) -> Vec<(&'a str, usize, usize)>
where
    T: Index<usize, Output = &'a str> + TraitParameterStorageUtils,
{
    impl_get_entities!(tags)
}

impl_get_entities!(
    &[&str],
    &[String],
    &[Cow<'_, str>],
    &[&Cow<'_, str>],
    &[Cow<'_, String>],
    &[&Cow<'_, String>],
    Vec<&str>,
    Vec<String>,
    Vec<Cow<'_, str>>,
    Vec<Cow<'_, String>>
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmes() {
        let example = vec!["B", "M", "E", "S"];
        let result = get_entities(&example);
        assert_eq!(result, vec![("_", 0, 2), ("_", 3, 3)]);

        let example = vec![
            "B".to_string(),
            "I".to_string(),
            "M".to_string(),
            "E".to_string(),
            "S".to_string(),
        ];
        let result = example.get_entities();
        // let result = get_entities(&example);
        assert_eq!(result, vec![("_", 0, 3), ("_", 4, 4)]);
    }

    #[test]
    fn test_get_entities() {
        let example = vec!["B-PER", "I-PER", "O", "B-LOC"];
        let result = get_entities(&example);
        assert_eq!(result, vec![("PER", 0, 1), ("LOC", 3, 3)]);
    }

    #[test]
    fn test_get_complex_entities() {
        let example = vec!["B-PER-CPX", "I-PER-CPX", "O", "B-LOC-CPX"];
        let result = get_entities(&example);
        assert_eq!(result, vec![("PER-CPX", 0, 1), ("LOC-CPX", 3, 3)]);
    }

    #[test]
    fn test_get_complex_with_o_end_entities() {
        let example = vec!["B-PER-CPX", "I-PER-CPX", "O", "B-LOC-CPX", "O"];
        let result = get_entities(&example);
        assert_eq!(result, vec![("PER-CPX", 0, 1), ("LOC-CPX", 3, 3)]);
    }
}
