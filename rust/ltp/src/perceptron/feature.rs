use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

pub trait TraitFeature {
    fn get_with_key(&self, key: &str) -> Option<usize>;
    fn get_vector_str(&self, key: &[&str]) -> Vec<usize> {
        key.iter()
            .map(|k| self.get_with_key(k))
            .into_iter()
            .flatten()
            .collect()
    }
    fn get_vector_string(&self, key: &[String]) -> Vec<usize> {
        key.iter()
            .map(|k| self.get_with_key(k))
            .into_iter()
            .flatten()
            .collect()
    }
}
pub trait TraitFeatureCompressUtils: Default + IntoIterator<Item = (String, usize)> {
    fn features(self) -> Vec<(String, usize)>;
}

impl<T> TraitFeatureCompressUtils for T
where
    T: Default + IntoIterator<Item = (String, usize)>,
{
    fn features(self) -> Vec<(String, usize)> {
        self.into_iter().collect()
    }
}

pub trait TraitFeaturesTrainUtils: Clone {
    fn feature_num(&self) -> usize;
    fn insert_feature(&mut self, key: String, value: usize);
    fn remove_feature(&mut self, key: &str) -> Option<usize>;
    fn put_feature(&mut self, key: String, value: usize);
    fn del_feature(&mut self, key: &str) -> Option<usize>;
}

impl<T> TraitFeature for &T
where
    T: TraitFeature,
{
    fn get_with_key(&self, key: &str) -> Option<usize> {
        self.deref().get_with_key(key)
    }
}

impl<T> TraitFeature for Arc<T>
where
    T: TraitFeature,
{
    fn get_with_key(&self, key: &str) -> Option<usize> {
        self.deref().get_with_key(key)
    }
}

impl<T> TraitFeaturesTrainUtils for &T
where
    T: TraitFeaturesTrainUtils,
{
    fn feature_num(&self) -> usize {
        self.deref().feature_num()
    }

    fn insert_feature(&mut self, key: String, value: usize) {
        self.deref().put_feature(key, value)
    }

    fn remove_feature(&mut self, key: &str) -> Option<usize> {
        self.deref().del_feature(key)
    }

    fn put_feature(&mut self, key: String, value: usize) {
        self.deref().insert_feature(key, value)
    }

    fn del_feature(&mut self, key: &str) -> Option<usize> {
        self.deref().remove_feature(key)
    }
}

impl<T> TraitFeaturesTrainUtils for Arc<T>
where
    T: TraitFeaturesTrainUtils,
{
    fn feature_num(&self) -> usize {
        self.deref().feature_num()
    }

    fn insert_feature(&mut self, key: String, value: usize) {
        self.deref().put_feature(key, value)
    }

    fn remove_feature(&mut self, key: &str) -> Option<usize> {
        self.deref().del_feature(key)
    }

    fn put_feature(&mut self, key: String, value: usize) {
        self.deref().insert_feature(key, value)
    }

    fn del_feature(&mut self, key: &str) -> Option<usize> {
        self.deref().remove_feature(key)
    }
}

// HashMap

impl TraitFeature for HashMap<String, usize> {
    fn get_with_key(&self, key: &str) -> Option<usize> {
        self.get(key).copied()
    }
}

impl TraitFeaturesTrainUtils for HashMap<String, usize> {
    fn feature_num(&self) -> usize {
        self.len()
    }

    fn insert_feature(&mut self, key: String, value: usize) {
        self.insert(key, value);
    }

    fn remove_feature(&mut self, key: &str) -> Option<usize> {
        self.remove(key)
    }

    fn put_feature(&mut self, key: String, value: usize) {
        self.insert(key, value);
    }

    fn del_feature(&mut self, key: &str) -> Option<usize> {
        self.remove(key)
    }
}
