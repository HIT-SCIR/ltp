use crate::perceptron::model::PaMode;
use crate::perceptron::{
    Definition, Perceptron, Sample, TraitFeature, TraitFeatureCompressUtils,
    TraitFeaturesTrainUtils, TraitParameter, TraitParameterStorage,
    TraitParameterStorageCompressUtils, TraitParameterStorageTrainUtils,
};
use anyhow::Result;
use num_traits::Float;
use rand::prelude::SliceRandom;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::thread;

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Algorithm<Param: TraitParameter> {
    AP(usize),
    PA(PaMode<Param>),
}

impl<Param: TraitParameter + Display> Display for Algorithm<Param> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::AP(threads) => {
                write!(f, "algorithm: AP (threads={})", threads)
            }
            Algorithm::PA(PaMode::Pa) => {
                write!(f, "algorithm: Pa")
            }
            Algorithm::PA(PaMode::PaI(c)) => {
                write!(f, "algorithm: PaI(c={})", c)
            }
            Algorithm::PA(PaMode::PaII(c)) => {
                write!(f, "algorithm: PaII(c={})", c)
            }
        }
    }
}

impl<Param: TraitParameter> Default for Algorithm<Param> {
    fn default() -> Self {
        Algorithm::AP(1)
    }
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone)]
pub struct Trainer<Define, Param = f64>
    where
        Define: Definition,
        Param: TraitParameter + Display,
{
    pub definition: Define,
    pub epoch: usize,
    pub shuffle: bool,
    pub verbose: bool,
    pub eval_threads: usize,

    // 训练算法
    pub algorithm: Algorithm<Param>,

    // 模型压缩参数
    pub compress: bool,
    pub ratio: f64,
    pub threshold: Param,

    pub train_set: Option<Vec<Sample>>,
    pub eval_set: Option<Vec<Sample>>,
}

macro_rules! impl_set_param {
    ($name:ident, $type:ty) => {
        pub fn $name(mut self, $name: $type) -> Self {
            self.$name = $name;
            self
        }
    };
}

impl<Define, Param> Trainer<Define, Param>
    where
        Param: TraitParameter + Display + Sync + Send + 'static,
        Define: Definition + Sync + Send + 'static,
{
    pub fn new() -> Self {
        Self {
            epoch: 1,
            shuffle: true,
            verbose: true,
            eval_threads: 8,
            compress: true,
            ratio: 0.3,
            threshold: Param::from(1e-3).unwrap_or_else(Param::one),
            ..Default::default()
        }
    }

    pub fn new_with_define(define: Define) -> Self {
        Self {
            epoch: 1,
            shuffle: true,
            verbose: true,
            eval_threads: 8,
            compress: true,
            ratio: 0.3,
            threshold: Param::from(1e-3).unwrap_or_else(Param::one),
            definition: define,
            ..Default::default()
        }
    }

    impl_set_param!(definition, Define);
    impl_set_param!(epoch, usize);
    impl_set_param!(shuffle, bool);
    impl_set_param!(verbose, bool);
    impl_set_param!(eval_threads, usize);
    impl_set_param!(compress, bool);
    impl_set_param!(ratio, f64);
    impl_set_param!(threshold, Param);
    impl_set_param!(algorithm, Algorithm<Param>);

    pub fn load_dataset<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Sample>> {
        let file = File::open(path)?;
        let dataset = self.definition.parse_gold_features(file)?;
        Ok(dataset)
    }

    pub fn train_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let dataset = self.load_dataset(path)?;
        self.train_set = Some(dataset);
        Ok(self)
    }

    pub fn eval_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let dataset = self.load_dataset(path)?;
        self.eval_set = Some(dataset);
        Ok(self)
    }

    pub fn display(self) -> Self {
        println!("{}", self);
        self
    }

    pub fn evaluate<Feature, ParamStorage>(
        &self,
        model: &Perceptron<Define, Feature, ParamStorage, Param>,
    ) -> Result<(f64, f64, f64)>
        where
            Feature: TraitFeature,
            Param: TraitParameter,
            ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageTrainUtils<Param>,
            Define: Definition,
    {
        if let Some(eval_set) = &self.eval_set {
            #[cfg(feature = "parallel")]
                let result = {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(self.eval_threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    eval_set
                        .par_iter()
                        .map(|(feature, labels)| model.evaluate(feature, labels))
                        .reduce_with(
                            |(a_correct, a_preds, a_labels), (b_correct, b_preds, b_labels)| {
                                (
                                    a_correct + b_correct,
                                    a_preds + b_preds,
                                    a_labels + b_labels,
                                )
                            },
                        )
                })
            };
            #[cfg(not(feature = "parallel"))]
                let result = eval_set
                .iter()
                .map(|(feature, labels)| model.evaluate(feature, labels))
                .reduce(
                    |(a_correct, a_preds, a_labels), (b_correct, b_preds, b_labels)| {
                        (
                            a_correct + b_correct,
                            a_preds + b_preds,
                            a_labels + b_labels,
                        )
                    },
                );

            if let Some((correct, preds_total, labels_total)) = result {
                let precision = correct as f64 / preds_total as f64;
                let recall = correct as f64 / labels_total as f64;
                let f1 = 2.0 * precision * recall / (precision + recall);
                return Ok((precision, recall, f1));
            }
        }
        Ok((0.0, 0.0, 0.0))
    }

    pub fn build<Feature, ParamStorage>(
        &self,
    ) -> Result<Perceptron<Define, Feature, ParamStorage, Param>>
        where
            ParamStorage: TraitParameterStorage<Param>
            + TraitParameterStorageTrainUtils<Param>
            + TraitParameterStorageCompressUtils<Param>
            + Send
            + Sync
            + 'static,
            Feature: TraitFeature
            + TraitFeaturesTrainUtils
            + TraitFeatureCompressUtils
            + ToOwned<Owned=Feature>
            + Send
            + Sync
            + 'static,
    {
        let mut features_set = HashSet::new();
        if let Some(train_set) = &self.train_set {
            for (sentence_features, _sentence_labels) in train_set {
                for word_features in sentence_features {
                    for features in word_features {
                        features_set.insert(features.to_owned());
                    }
                }
            }
        }

        let bias = if self.definition.use_viterbi() {
            // transition part of viterbi
            self.definition.label_num()
        } else {
            0
        };

        let mut features = Feature::default();
        for (feature, id) in features_set
            .into_iter()
            .enumerate()
            .map(|(idx, feature)| (feature, idx + bias))
        {
            features.insert_feature(feature, id);
        }

        let model = match &self.algorithm {
            Algorithm::AP(threads) => {
                let threads = *threads;
                if threads <= 1 {
                    self.build_ap(features)?
                } else {
                    self.build_ap_parallel(features, threads)?
                }
            }
            Algorithm::PA(mode) => self.build_pa(features, mode)?,
        };

        let model = if self.compress {
            let model = model.compress(self.ratio, self.threshold);
            let (p, r, f1) = self.evaluate(&model)?;

            println!("Compressed: precision: {}, recall: {}, f1: {}", p, r, f1);
            model
        } else {
            model
        };

        Ok(model)
    }

    pub fn build_ap<Feature, ParamStorage>(
        &self,
        features: Feature,
    ) -> Result<Perceptron<Define, Feature, ParamStorage, Param>>
        where
            ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageTrainUtils<Param>,
            Feature: TraitFeature + TraitFeaturesTrainUtils,
    {
        let label_num = self.definition.label_num();
        let bias = if self.definition.use_viterbi() {
            // transition part of viterbi
            label_num * label_num
        } else {
            0
        };
        let features_num = features.feature_num();
        let parameters_len = bias + features_num * label_num;

        let parameters = ParamStorage::init(Param::zero(), parameters_len);

        let mut perceptron =
            Perceptron::new_with_parameters(self.definition.clone(), features, parameters);

        let mut best_f1 = f64::neg_infinity();
        let mut best_parameters = ParamStorage::default();

        if let Some(train_set) = &self.train_set {
            let mut rng = rand::thread_rng();
            let mut current = 0;
            let mut total = vec![Param::zero(); parameters_len];
            let mut timestamp = vec![0; parameters_len];
            let mut train_set = train_set.clone();
            for epoch in 0..self.epoch {
                if self.shuffle {
                    train_set.shuffle(&mut rng);
                }
                for (feature, labels) in train_set.iter() {
                    current += 1;
                    perceptron.ap_train_iter(feature, labels, &mut total, &mut timestamp, current);
                }

                let backup = perceptron.parameters.clone();
                perceptron.average(&total, &timestamp, current);

                let (p, r, f1) = self.evaluate(&perceptron)?;

                println!(
                    "epoch: {}, precision: {}, recall: {}, f1: {}",
                    epoch, p, r, f1
                );

                if f1 > best_f1 {
                    best_f1 = f1;
                    best_parameters = perceptron.parameters;
                }
                perceptron.parameters = backup;
            }
        }
        perceptron.parameters = best_parameters;
        Ok(perceptron)
    }

    pub fn build_pa<Feature, ParamStorage>(
        &self,
        features: Feature,
        pa_mode: &PaMode<Param>,
    ) -> Result<Perceptron<Define, Feature, ParamStorage, Param>>
        where
            ParamStorage: TraitParameterStorage<Param> + TraitParameterStorageTrainUtils<Param>,
            Feature: TraitFeature + TraitFeaturesTrainUtils,
    {
        let label_num = self.definition.label_num();
        let bias = if self.definition.use_viterbi() {
            // transition part of viterbi
            label_num * label_num
        } else {
            0
        };
        let features_num = features.feature_num();
        let parameters_len = bias + features_num * label_num;

        let parameters = ParamStorage::init(Param::zero(), parameters_len);
        let mut perceptron =
            Perceptron::new_with_parameters(self.definition.clone(), features, parameters);

        let mut best_f1 = f64::neg_infinity();
        let mut best_parameters = ParamStorage::default();

        if let Some(train_set) = &self.train_set {
            let mut rng = rand::thread_rng();
            let mut current = 0;
            let mut total = vec![Param::zero(); parameters_len];
            let mut timestamp = vec![0; parameters_len];
            let mut train_set = train_set.clone();
            for epoch in 0..self.epoch {
                if self.shuffle {
                    train_set.shuffle(&mut rng);
                }
                for (feature, labels) in train_set.iter() {
                    current += 1;
                    perceptron.pa_train_iter(
                        feature,
                        labels,
                        &mut total,
                        &mut timestamp,
                        current,
                        pa_mode,
                    );
                }

                let backup = perceptron.parameters.clone();
                perceptron.average(&total, &timestamp, current);

                let (p, r, f1) = self.evaluate(&perceptron)?;

                println!(
                    "epoch: {}, precision: {}, recall: {}, f1: {}",
                    epoch, p, r, f1
                );

                if f1 > best_f1 {
                    best_f1 = f1;
                    best_parameters = perceptron.parameters;
                }
                perceptron.parameters = backup;
            }
        }
        perceptron.parameters = best_parameters;
        Ok(perceptron)
    }

    pub fn build_ap_parallel<Feature, ParamStorage>(
        &self,
        features: Feature,
        threads: usize,
    ) -> Result<Perceptron<Define, Feature, ParamStorage, Param>>
        where
            ParamStorage: TraitParameterStorage<Param>
            + TraitParameterStorageTrainUtils<Param>
            + Send
            + Sync
            + 'static,
            Feature: TraitFeature
            + TraitFeaturesTrainUtils
            + TraitFeatureCompressUtils
            + ToOwned<Owned=Feature>
            + Send
            + Sync
            + 'static,
    {
        let features = Arc::new(features);

        let label_num = self.definition.label_num();
        let bias = if self.definition.use_viterbi() {
            // transition part of viterbi
            label_num * label_num
        } else {
            0
        };
        let features_num = features.feature_num();
        let parameters_len = bias + features_num * label_num;

        let mut best_f1 = f64::neg_infinity();
        let mut best_parameters = ParamStorage::default();
        let mut parameters = vec![ParamStorage::init(Param::zero(), parameters_len); threads];

        if let Some(train_set) = &self.train_set {
            let chunk_size = (train_set.len() as f64 / threads as f64) as usize + 1;
            let train_set = Arc::new(RwLock::new(train_set.clone()));

            for epoch in 0..self.epoch {
                if self.shuffle {
                    let mut rng = rand::thread_rng();
                    train_set.write().unwrap().shuffle(&mut rng);
                }
                let mut children = vec![];
                for thread in 0..threads {
                    let clone_feature = Arc::clone(&features);
                    let train_set_clone = Arc::clone(&train_set);
                    let definition = self.definition.clone();
                    let parameters = parameters.pop().unwrap();
                    children.push(thread::spawn(move || -> ParamStorage {
                        let mut perceptron =
                            Perceptron::new_with_parameters(definition, clone_feature, parameters);
                        let shared = train_set_clone.read().unwrap();
                        if let Some(chunk) = shared.chunks(chunk_size).nth(thread) {
                            for (feature, labels) in chunk {
                                perceptron.ap_train_parallel_iter(feature, labels);
                            }
                        };
                        perceptron.parameters
                    }));
                }

                for child in children {
                    parameters.push(child.join().unwrap());
                }

                let mut mean_parameters = ParamStorage::init(Param::zero(), parameters[0].len());

                for j in 0..mean_parameters.len() {
                    for parameters_i_thread in parameters.iter() {
                        mean_parameters[j] += parameters_i_thread[j];
                    }
                    mean_parameters[j] /= Param::from(threads).unwrap();
                    for parameters_i_thread in &mut parameters {
                        parameters_i_thread[j] = mean_parameters[j];
                    }
                }

                let clone_feature = Arc::clone(&features);
                let perceptron = Perceptron::new_with_parameters(
                    self.definition.clone(),
                    clone_feature.deref(),
                    mean_parameters,
                );

                let (p, r, f1) = self.evaluate(&perceptron)?;
                println!(
                    "epoch: {}, precision: {}, recall: {}, f1: {}",
                    epoch, p, r, f1
                );
                if f1 > best_f1 {
                    best_parameters = perceptron.parameters;
                    best_f1 = f1;
                }
            }
        }

        let features = features.deref();
        let features: Feature = features.to_owned();

        let perceptron =
            Perceptron::new_with_parameters(self.definition.clone(), features, best_parameters);

        Ok(perceptron)
    }
}

impl<Define, Param> Display for Trainer<Define, Param>
    where
        Define: Definition,
        Param: TraitParameter + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Trainer {{")?;
        writeln!(f, "  epoch: {}", self.epoch)?;
        writeln!(f, "  shuffle: {}", self.shuffle)?;
        writeln!(f, "  verbose: {}", self.verbose)?;
        writeln!(f, "  {}", self.algorithm)?;
        writeln!(f, "  eval_threads: {}", self.eval_threads)?;

        if self.compress {
            writeln!(
                f,
                "  compress: {{ ratio: {} threshold: {} }}",
                self.ratio, self.threshold
            )?;
        }
        if let Some(train_set) = &self.train_set {
            writeln!(f, "  train_set: {}", train_set.len())?;
        }
        if let Some(eval_set) = &self.eval_set {
            writeln!(f, "  eval_set: {}", eval_set.len())?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}
