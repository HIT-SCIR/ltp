use crate::perceptron::definition::CommonDefinePredict;
use crate::perceptron::GenericItem;
use crate::perceptron::{
    Definition, TraitFeature, TraitFeatureCompressUtils, TraitFeaturesTrainUtils, TraitParameter,
    TraitParameterStorage, TraitParameterStorageCompressUtils, TraitParameterStorageTrainUtils,
};
use anyhow::Result;
use binary_heap_plus::BinaryHeap;
use itertools::Itertools;
use num_traits::NumCast;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::iter::zip;
use std::mem::swap;

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaMode<Param>
where
    Param: TraitParameter,
{
    Pa,
    PaI(Param),
    PaII(Param),
}

impl<Param: TraitParameter> Default for PaMode<Param> {
    fn default() -> Self {
        PaMode::Pa
    }
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone)]
pub struct Perceptron<Define, Feature, ParamStorage, Param>
where
    Define: Definition,
    Feature: TraitFeature,
    ParamStorage: TraitParameterStorage<Param>,
    Param: TraitParameter,
{
    pub definition: Define,
    pub features: Feature,
    pub parameters: ParamStorage,
    #[cfg_attr(feature = "serialization", serde(skip_serializing))]
    __phantom: Option<Param>,
}

impl<Define, Feature, ParamStorage, Param> Display
    for Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
    Define: Definition,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Perceptron [ Num of Params: {} Precision: {} ]",
            self.parameters.len(),
            std::any::type_name::<Param>()
        )
    }
}

unsafe impl<Define, Feature, ParamStorage, Param> Send
    for Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
    Define: Definition,
{
}

unsafe impl<Define, Feature, ParamStorage, Param> Sync
    for Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
    Define: Definition,
{
}

impl<Define, Feature, ParamStorage, Param> Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
    Define: Definition,
{
    pub fn new_with_parameters(
        definition: Define,
        features: Feature,
        parameters: ParamStorage,
    ) -> Self {
        Perceptron {
            features,
            parameters,
            definition,
            __phantom: Default::default(),
        }
    }

    pub fn display(&self) -> String {
        format!("{}", self)
    }

    // 通用部分
    fn score_base(&self, features: &Vec<usize>, label: usize) -> Param {
        let label_num = self.definition.label_num();
        let mut score = Param::zero();
        for &feature in features {
            score += self.parameters[feature * label_num + label];
        }
        score
    }

    fn viterbi_decode(&self, features: &[Vec<usize>]) -> Vec<usize> {
        let label_num = self.definition.label_num();
        let mut pre_matrix = vec![0usize; features.len() * label_num];
        let mut score_last = vec![Param::zero(); label_num];
        let mut score_now = vec![Param::zero(); label_num];

        let first_feature = &features[0];
        for label_idx in 0..label_num {
            pre_matrix[label_idx] = label_idx;
            score_last[label_idx] = self.score_base(first_feature, label_idx);
        }

        for (i, feature) in features.iter().enumerate().skip(1) {
            let base = i * label_num;

            for label_idx in 0..label_num {
                let mut max_score = Param::min_value();
                let score_base = self.score_base(feature, label_idx);

                for (pre_label_idx, &last_score) in score_last.iter().enumerate() {
                    // transition
                    let transition_score = self.parameters[pre_label_idx * label_num + label_idx];
                    // let transition_score = Param::zero();
                    let score = last_score + score_base + transition_score;
                    if score > max_score {
                        max_score = score;
                        pre_matrix[base + label_idx] = pre_label_idx;
                        score_now[label_idx] = max_score;
                    }
                }
            }
            swap(&mut score_last, &mut score_now);
        }

        let mut max_score_idx = score_last
            .iter()
            .position_max_by(|x, y| match x.partial_cmp(y) {
                None => Ordering::Equal,
                Some(order) => order,
            })
            .unwrap();

        let mut res = vec![0; features.len()];

        for i in (0..features.len()).rev() {
            let label_idx = max_score_idx;
            res[i] = label_idx;
            max_score_idx = pre_matrix[i * label_num + label_idx];
        }

        res
    }

    fn simple_decode(&self, features: &[Vec<usize>]) -> Vec<usize> {
        let label_num = self.definition.label_num();
        let mut res = vec![0; features.len()];

        for (i, feature) in features.iter().enumerate() {
            let mut max_score = Param::min_value();
            for label_idx in 0..label_num {
                let score = self.score_base(feature, label_idx);
                if score > max_score {
                    max_score = score;
                    res[i] = label_idx;
                }
            }
        }

        res
    }

    // viterbi decode
    pub fn decode(&self, features: &[Vec<usize>]) -> Vec<usize> {
        if self.definition.use_viterbi() {
            self.viterbi_decode(features)
        } else {
            self.simple_decode(features)
        }
    }

    pub fn evaluate(&self, inputs: &[Vec<String>], labels: &[usize]) -> (usize, usize, usize) {
        let features: Vec<_> = inputs
            .iter()
            .map(|f| self.features.get_vector_string(f))
            .collect();
        let preds = self.decode(&features);
        self.definition.evaluate(&preds, labels)
    }
}

impl<Define, Feature, ParamStorage, Param> Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
    Define: Definition + CommonDefinePredict,
{
    pub fn predict_with_buffer(
        &self,
        sentence: <Define::RawFeature as GenericItem>::Item,
        buffer: &mut Vec<u8>,
    ) -> Result<<Define::Prediction as GenericItem>::Item> {
        let (fragment, features) = self
            .definition
            .parse_features_with_buffer(&sentence, buffer)?;
        let features: Vec<_> = features
            .iter()
            .map(|f| self.features.get_vector_str(f))
            .collect();
        let preds = self.decode(&features);

        Ok(self.definition.predict(&sentence, &fragment, &preds))
    }
}

impl<Feature, ParamStorage, Param> Perceptron<POSDefinition, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
{
    pub fn predict(&self, sentence: &[&str]) -> Result<Vec<&str>> {
        let mut buffer = Vec::with_capacity(sentence.len() * 180);
        self.predict_with_buffer(sentence, &mut buffer)
    }
}

impl<Feature, ParamStorage, Param> Perceptron<NERDefinition, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
{
    pub fn predict(&self, sentence: (&[&str], &[&str])) -> Result<Vec<&str>> {
        let mut buffer = Vec::with_capacity(sentence.0.len() * 150);
        self.predict_with_buffer(sentence, &mut buffer)
    }
}

use crate::{get_entities, CWSDefinition, NERDefinition, POSDefinition};

impl<Feature, ParamStorage, Param> Perceptron<CWSDefinition, Feature, ParamStorage, Param>
where
    Feature: TraitFeature,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param>,
{
    pub fn check_feature(&self, feature: &str) -> Option<usize> {
        self.features.get_with_key(feature)
    }

    pub fn predict<'a>(&self, sentence: &'a str) -> Result<Vec<&'a str>> {
        let mut buffer = Vec::with_capacity(sentence.len() * 20);
        self.predict_with_buffer(sentence, &mut buffer)
    }

    pub fn predict_with_buffer<'a>(
        &self,
        sentence: &'a str,
        buffer: &mut Vec<u8>,
    ) -> Result<Vec<&'a str>> {
        let (fragments, features) = self
            .definition
            .parse_features_with_buffer(&sentence, buffer)?;
        let features: Vec<_> = features
            .iter()
            .map(|f| self.features.get_vector_str(f))
            .collect();
        let preds = self.decode(&features);

        let preds = self.definition.to_labels(&preds);
        let preds = get_entities(&preds);
        Ok(preds
            .into_iter()
            .map(|(_, start, end)| {
                let start = fragments[start];
                let end = fragments[end + 1];
                &sentence[start..end]
            })
            .collect::<Vec<_>>())
    }
}

impl<Feature, ParamStorage, Param> Perceptron<CWSDefinition, Feature, ParamStorage, Param>
where
    Feature: TraitFeature + TraitFeaturesTrainUtils,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageCompressUtils<Param>,
{
    pub fn add_core_rule(&mut self, key: &str, s: Param, b: Param, m: Param, e: Param) {
        if self.features.get_with_key(key).is_none() {
            let feature_num = self.parameters.len() / 4;
            self.features.insert_feature(String::from(key), feature_num);

            // S B M E
            self.parameters.push(s);
            self.parameters.push(b);
            self.parameters.push(m);
            self.parameters.push(e);
        }
    }

    pub fn enable_feature_rule(&mut self, key: &str, feature: &str) {
        if let Some(index) = self.features.get_with_key(key) {
            self.features.insert_feature(feature.to_string(), index);
        }
    }

    pub fn disable_feature_rule(&mut self, feature: &str) -> Option<usize> {
        self.features.remove_feature(feature)
    }
}

// 模型训练
impl<Define, Feature, ParamStorage, Param> Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature + TraitFeaturesTrainUtils,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageTrainUtils<Param>,
    Define: Definition,
{
    // 被动攻击算法
    pub fn pa_train_iter(
        &mut self,
        inputs: &[Vec<String>],
        labels: &[usize],
        total: &mut [Param],
        timestamp: &mut [usize],
        current: usize,
        mode: &PaMode<Param>,
    ) {
        let label_num = self.definition.label_num();
        let features: Vec<_> = inputs
            .iter()
            .map(|f| self.features.get_vector_string(f))
            .collect();
        let preds = self.decode(&features);

        if labels.ne(&preds) {
            let errors = zip(labels, &preds)
                .enumerate()
                .filter(|&(_, (gold, pred))| gold != pred);

            for (idx, (gold, pred)) in errors {
                let mut score = Param::zero(); // W \dot X
                let mut norm = Param::zero(); // ||x||^2

                if self.definition.use_viterbi() && idx > 0 {
                    score += self.parameters[labels[idx - 1] * label_num + gold];
                    score -= self.parameters[preds[idx - 1] * label_num + pred];

                    norm += Param::one() + Param::one();
                    // norm += Param::one().powi(2) + Param::one().powi(2);
                }

                features[idx].iter().for_each(|&feat| {
                    score += self.parameters[feat * label_num + gold];
                    score -= self.parameters[feat * label_num + pred];
                    norm += Param::one() + Param::one();
                    // norm += Param::one().powi(2) + Param::one().powi(2);
                });

                if norm < Param::from(1e-8).unwrap() {
                    continue;
                }

                let step = match mode {
                    PaMode::Pa => (Param::one() - score) / norm,
                    PaMode::PaI(c) => {
                        let step: Param = (Param::one() - score) / norm;
                        match step.partial_cmp(c) {
                            None => Param::zero(),
                            Some(Ordering::Less) => step,
                            Some(Ordering::Equal) => step,
                            Some(Ordering::Greater) => *c,
                        }
                    }
                    PaMode::PaII(c) => {
                        (Param::one() - score)
                            / (norm + Param::one() / (Param::from(2).unwrap() * *c))
                    }
                };

                features[idx].iter().for_each(|&feat| {
                    self.record(feat * label_num + gold, step, total, timestamp, current);
                    if pred < &self.parameters.len() {
                        self.record(feat * label_num + pred, -step, total, timestamp, current);
                    }
                });

                if self.definition.use_viterbi() && idx > 0 {
                    // transition
                    self.record(
                        labels[idx - 1] * label_num + gold,
                        step,
                        total,
                        timestamp,
                        current,
                    );
                    self.record(
                        preds[idx - 1] * label_num + pred,
                        -step,
                        total,
                        timestamp,
                        current,
                    );
                }
            }
        }
    }

    // 单线程 averaged perceptron 算法
    pub fn average(&mut self, total: &[Param], timestamp: &[usize], current: usize) {
        for feat in 0..self.parameters.len() {
            let passed: Param = NumCast::from(current - timestamp[feat]).unwrap();
            let total_step = total[feat] + passed * self.parameters[feat];
            self.parameters[feat] = total_step / NumCast::from(current).unwrap();
        }
    }

    fn record(
        &mut self,
        feat: usize,
        value: Param,
        total: &mut [Param],
        timestamp: &mut [usize],
        current: usize,
    ) {
        let passed = current - timestamp[feat];
        total[feat] += self.parameters[feat] * NumCast::from(passed).unwrap();
        timestamp[feat] = current;

        self.parameters[feat] += value;
    }

    pub fn ap_train_iter(
        &mut self,
        inputs: &[Vec<String>],
        labels: &[usize],
        total: &mut [Param],
        timestamp: &mut [usize],
        current: usize,
    ) {
        let label_num = self.definition.label_num();
        let features: Vec<_> = inputs
            .iter()
            .map(|f| self.features.get_vector_string(f))
            .collect();
        let preds = self.decode(&features);

        if labels.ne(&preds) {
            for (idx, (&gold, &pred)) in zip(labels, &preds)
                .enumerate()
                .filter(|&(_, (gold, pred))| gold != pred)
            {
                features[idx].iter().for_each(|&feat| {
                    self.record(
                        feat * label_num + gold,
                        Param::one(),
                        total,
                        timestamp,
                        current,
                    );
                    if pred < self.parameters.len() {
                        self.record(
                            feat * label_num + pred,
                            -Param::one(),
                            total,
                            timestamp,
                            current,
                        );
                    }
                });

                if self.definition.use_viterbi() && idx > 0 {
                    // transition
                    self.record(
                        labels[idx - 1] * label_num + gold,
                        Param::one(),
                        total,
                        timestamp,
                        current,
                    );
                    self.record(
                        preds[idx - 1] * label_num + pred,
                        -Param::one(),
                        total,
                        timestamp,
                        current,
                    );
                }
            }
        }
    }

    // 并行 averaged perceptron 算法
    pub fn ap_train_parallel_iter(&mut self, inputs: &[Vec<String>], labels: &[usize]) {
        let label_num = self.definition.label_num();
        let features: Vec<_> = inputs
            .iter()
            .map(|f| self.features.get_vector_string(f))
            .collect();
        let preds = self.decode(&features);

        if labels.ne(&preds) {
            for (idx, (&gold, &pred)) in zip(labels, &preds)
                .enumerate()
                .filter(|&(_, (gold, pred))| gold != pred)
            {
                features[idx].iter().for_each(|&feat| {
                    self.parameters[feat * label_num + gold] += Param::one();
                    if pred < self.parameters.len() {
                        self.parameters[feat * label_num + pred] -= Param::one();
                    }
                });

                if self.definition.use_viterbi() && idx > 0 {
                    // transition
                    self.parameters[labels[idx - 1] * label_num + gold] += Param::one();
                    self.parameters[preds[idx - 1] * label_num + pred] -= Param::one();
                }
            }
        }
    }
}

// 模型压缩
impl<Define, Feature, ParamStorage, Param> Perceptron<Define, Feature, ParamStorage, Param>
where
    Feature: TraitFeature + TraitFeaturesTrainUtils + TraitFeatureCompressUtils,
    Param: TraitParameter,
    ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageCompressUtils<Param>,
    Define: Definition,
{
    fn param_score(parameters: &ParamStorage, label_num: usize, feature: usize) -> Param {
        let mut score = Param::zero();
        let start = feature * label_num;
        let end = start + label_num;
        for i in start..end {
            score += parameters[i].abs();
        }
        score
    }
    pub fn compress(self, ratio: f64, threshold: Param) -> Self {
        assert!(0.0 < ratio && ratio <= 1.0, "压缩比必须介于 0 和 1 之间");
        let label_num = self.definition.label_num();
        let old_features = self.features.features();
        let old_parameters = self.parameters;

        let mut filter_set: HashSet<usize> = HashSet::new();
        let target_features_num = ((old_features.len() / label_num) as f64 * ratio) as usize;

        for (_feature, parm_idx) in &old_features {
            let score = Self::param_score(&old_parameters, label_num, *parm_idx);
            if score < threshold {
                filter_set.insert(*parm_idx);
            }
        }

        if filter_set.len() < target_features_num {
            let mut heap = BinaryHeap::new_by(|&a: &(usize, Param), &b: &(usize, Param)| {
                a.1.partial_cmp(&b.1).unwrap()
            });
            for (_feature, parm_idx) in old_features.iter() {
                let score = Self::param_score(&old_parameters, label_num, *parm_idx);
                if score >= threshold {
                    heap.push((*parm_idx, score));
                }
            }

            while filter_set.len() < target_features_num {
                let (param, _) = heap.pop().unwrap();
                filter_set.insert(param);
            }
        }

        let (mut new_parameters, bias) = if self.definition.use_viterbi() {
            // transition part of viterbi
            let trans_len = label_num * label_num;
            let mut new_parameters = ParamStorage::with_capacity(
                (old_features.len() - filter_set.len()) * label_num + trans_len,
            );

            for i in 0..trans_len {
                new_parameters.push(old_parameters[i]);
            }

            (new_parameters, label_num)
        } else {
            (
                ParamStorage::with_capacity((old_features.len() - filter_set.len()) * label_num),
                0,
            )
        };

        let mut new_features: Feature = Default::default();
        for (idx, (feature, old_idx)) in old_features
            .into_iter()
            .filter(|(_, p)| !filter_set.contains(p))
            .enumerate()
        {
            new_features.insert_feature(feature, idx + bias);
            let param_start = old_idx * label_num;
            for label_idx in 0..label_num {
                new_parameters.push(old_parameters[param_start + label_idx]);
            }
        }

        Self::new_with_parameters(self.definition, new_features, new_parameters)
    }
}
