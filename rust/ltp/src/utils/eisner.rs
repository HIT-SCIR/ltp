use num_traits::Float;
use std::cmp::{max, min};
use std::fmt::Display;

fn fill<T: Copy>(array: &mut [T], num: T, size: usize) {
    for item in array.iter_mut().take(size) {
        *item = num;
    }
}

fn backtrack<const C: bool>(
    p_i: &[usize],
    p_c: &[usize],
    i: usize,
    j: usize,
    blk_bias: usize,
    head: &mut [usize],
    remove_root: usize,
) {
    if i == j {
        return;
    }
    if C {
        let r = p_c[i * blk_bias + j];
        backtrack::<false>(p_i, p_c, i, r, blk_bias, head, remove_root);
        backtrack::<true>(p_i, p_c, r, j, blk_bias, head, remove_root);
    } else {
        let r = p_i[i * blk_bias + j];
        head[j - remove_root] = i;
        backtrack::<true>(p_i, p_c, min(i, j), r, blk_bias, head, remove_root);
        backtrack::<true>(p_i, p_c, max(i, j), r + 1, blk_bias, head, remove_root);
    }
}

pub fn eisner<T>(scores: &[T], stn_length: &[usize], remove_root: bool) -> Vec<Vec<usize>>
where
    T: Float + Display,
{
    // scores [b, w, n]
    let batch = stn_length.len();
    let max_stn_len = *stn_length.iter().max().unwrap();
    let score_block_size = max_stn_len * max_stn_len;

    // [b, n, w]
    let mut bs_i = vec![T::neg_infinity(); score_block_size];
    let mut bs_c = vec![T::neg_infinity(); score_block_size];

    let mut bp_i = vec![0; score_block_size];
    let mut bp_c = vec![0; score_block_size];

    let remove_root = remove_root as usize;

    let mut res = Vec::new();
    for (b, &max_stn_len_use) in stn_length.iter().enumerate().take(batch) {
        fill(&mut bs_i, T::neg_infinity(), score_block_size);
        fill(&mut bs_c, T::neg_infinity(), score_block_size);
        fill(&mut bp_i, 0, score_block_size);
        fill(&mut bp_c, 0, score_block_size);

        let bscore_bias = b * score_block_size;

        for k in 0..max_stn_len_use {
            bs_i[k * max_stn_len_use + k] = T::zero();
            bs_c[k * max_stn_len_use + k] = T::zero();
        }

        for w in 1..max_stn_len_use {
            let n = max_stn_len_use - w;
            // I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_score = T::neg_infinity();
                let mut max_index = 0;
                for r in i..j {
                    let s = bs_c[i * max_stn_len_use + r]
                        + bs_c[j * max_stn_len_use + r + 1]
                        + scores[bscore_bias + i * max_stn_len + j];
                    if s > max_score {
                        max_score = s;
                        max_index = r;
                    }
                }
                bs_i[j * max_stn_len_use + i] = max_score;
                bp_i[j * max_stn_len_use + i] = max_index;
            }
            // I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[i * max_stn_len_use + r]
                        + bs_c[j * max_stn_len_use + r + 1]
                        + scores[bscore_bias + j * max_stn_len + i];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_i[i * max_stn_len_use + j] = max_score;
                bp_i[i * max_stn_len_use + j] = max_index;
            }
            // C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i..j {
                    let s = bs_c[r * max_stn_len_use + i] + bs_i[j * max_stn_len_use + r];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[j * max_stn_len_use + i] = max_score;
                bp_c[j * max_stn_len_use + i] = max_index;
            }
            // C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
            for i in 0..n {
                let j = i + w;
                let mut max_index = 0;
                let mut max_score = T::neg_infinity();
                for r in i + 1..j + 1 {
                    let s = bs_i[i * max_stn_len_use + r] + bs_c[r * max_stn_len_use + j];
                    if s > max_score {
                        max_index = r;
                        max_score = s;
                    }
                }
                bs_c[i * max_stn_len_use + j] = max_score;
                bp_c[i * max_stn_len_use + j] = max_index;
            }
            if stn_length[b] != w {
                // todo: check it
                // bs_c[0 * max_stn_len_use + w] = T::neg_infinity();
                bs_c[w] = T::neg_infinity();
            }
        }
        let mut b_head = vec![1usize; max_stn_len_use - remove_root];
        backtrack::<true>(
            &bp_i,
            &bp_c,
            0,
            max_stn_len_use - 1,
            max_stn_len_use,
            &mut b_head,
            remove_root,
        );
        res.push(b_head);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::eisner;
    use ndarray::{Array1, Array3};
    use ndarray_npy::{NpzReader, ReadNpzError};
    use std::fs::File;

    #[test]
    fn test_eisner() -> Result<(), ReadNpzError> {
        let mut npz = NpzReader::new(File::open("test/eisner.npz").unwrap())?;
        let scores: Array3<f32> = npz.by_name("scores.npy")?;
        let stn_length: Array1<i64> = npz.by_name("stn_length.npy")?;
        let correct: Array1<i64> = npz.by_name("correct.npy")?;

        let stn_length: Vec<usize> = stn_length.iter().map(|&x| x as usize).collect();
        let output = eisner(scores.as_slice().unwrap(), stn_length.as_slice(), true);

        let correct: Vec<usize> = correct.iter().map(|&x| x as usize).collect();
        let output: Vec<usize> = output.iter().flatten().map(|&x| x).collect();

        assert_eq!(correct, output);

        Ok(())
    }
}
