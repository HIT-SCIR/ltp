use num_traits::PrimInt;

pub fn viterbi_decode_postprocessing<T>(
    history: &[T],
    last_tags: &[T],
    stn_lengths: &[usize],
    labels_num: usize,
) -> Vec<Vec<T>>
where
    T: PrimInt,
{
    // history
    // max_stn_len * stn_num * labels_num
    let stn_num: usize = stn_lengths.iter().sum();
    let b_bias = stn_num * labels_num;
    let i_bias = labels_num;

    let mut result: Vec<Vec<T>> = Vec::new();
    let mut stn_idx = 0;
    for &stn_len in stn_lengths {
        for _search_idx in 0..stn_len {
            let best_last_tag = last_tags[stn_idx];
            let mut best_tags = vec![best_last_tag];

            // history
            // stn_len *  stn_num * labels_num
            for search_end in 1..(stn_len) {
                // last one has been used
                let search_end = (stn_len - 1) - search_end;
                let forward_best = *best_tags.last().unwrap();
                let index =
                    search_end * b_bias + stn_idx * i_bias + forward_best.to_usize().unwrap();
                let last_best = history[index];
                best_tags.push(last_best);
            }
            best_tags.reverse();
            result.push(best_tags);
            stn_idx += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::viterbi_decode_postprocessing;
    use ndarray::{Array1, Array3};
    use ndarray_npy::{NpzReader, ReadNpzError};
    use std::fs::File;

    #[test]
    fn test_viterbi() -> Result<(), ReadNpzError> {
        let mut npz = NpzReader::new(File::open("test/viterbi.npz").unwrap())?;
        let srl_history: Array3<i64> = npz.by_name("srl_history.npy")?;
        let srl_last_tags: Array1<i64> = npz.by_name("srl_last_tags.npy")?;
        let word_nums: Array1<i64> = npz.by_name("word_nums.npy")?;
        let correct: Array1<i64> = npz.by_name("correct.npy")?;

        let label_num = srl_history.dim().2;
        let word_nums: Vec<usize> = word_nums.iter().map(|&x| x as usize).collect();

        let output = viterbi_decode_postprocessing(
            srl_history.as_slice().unwrap(),
            srl_last_tags.as_slice().unwrap(),
            word_nums.as_slice(),
            label_num,
        );

        let correct: Vec<i64> = correct.iter().map(|&x| x).collect();
        let output: Vec<i64> = output.iter().flatten().map(|&x| x).collect();

        assert_eq!(correct, output);

        Ok(())
    }
}
