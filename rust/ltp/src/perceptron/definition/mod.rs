mod cws;
mod ner;
mod pos;

use std::collections::HashSet;
use std::fmt::Debug;
use std::io::Read;

use crate::get_entities;
use crate::perceptron::Sample;
pub use cws::CWSDefinition;
pub use ner::NERDefinition;
pub use pos::POSDefinition;

pub trait GenericItem<'a> {
    type Item;
}

pub trait Definition: Default + Debug + Clone {
    type Fragment: ?Sized + for<'any> GenericItem<'any>;
    type Prediction: ?Sized + for<'any> GenericItem<'any>;
    type RawFeature: ?Sized + for<'any> GenericItem<'any>;

    fn use_viterbi(&self) -> bool {
        false
    }
    fn labels(&self) -> Vec<String>;
    fn label_num(&self) -> usize;
    fn label_to(&self, label: &str) -> usize;
    fn to_label(&self, index: usize) -> &str;
    fn parse_features(
        &self,
        line: <Self::RawFeature as GenericItem>::Item,
    ) -> (<Self::Fragment as GenericItem>::Item, Vec<Vec<String>>);
    fn parse_gold_features<R: Read>(&self, reader: R) -> Vec<Sample>;
    fn to_labels(&self, index: &[usize]) -> Vec<&str> {
        index.iter().map(|&p| self.to_label(p)).collect()
    }
    fn predict(
        &self,
        fragments: &<Self::Fragment as GenericItem>::Item,
        preds: &[usize],
    ) -> <Self::Prediction as GenericItem>::Item;

    fn evaluate(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize);

    fn evaluate_tags(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize) {
        (
            predicts
                .iter()
                .zip(labels.iter())
                .map(|(p, l)| if p == l { 1usize } else { 0usize })
                .sum::<usize>(),
            predicts.len(),
            labels.len(),
        )
    }
    fn evaluate_entities(&self, predicts: &[usize], labels: &[usize]) -> (usize, usize, usize) {
        let predicts = self.to_labels(predicts);
        let labels = self.to_labels(labels);

        let predicts: HashSet<_> = get_entities(&predicts).into_iter().collect();
        let labels: HashSet<_> = get_entities(&labels).into_iter().collect();

        let correct = predicts.intersection(&labels).count();
        (correct, predicts.len(), labels.len())
    }
}