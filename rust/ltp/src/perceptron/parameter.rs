use num_traits::{Float, Num, NumAssignOps};
use std::ops::{Deref, Index, IndexMut};

pub trait TraitParameter: Float + NumAssignOps + Default {}

impl<T> TraitParameter for T where T: Float + NumAssignOps + Default {}

pub trait TraitParameterStorageUtils {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> TraitParameterStorageUtils for &T
where
    T: TraitParameterStorageUtils,
{
    fn len(&self) -> usize {
        self.deref().len()
    }
    fn is_empty(&self) -> bool {
        self.deref().is_empty()
    }
}

pub trait TraitParameterStorage<Param>:
    Index<usize, Output = Param> + TraitParameterStorageUtils
where
    Param: TraitParameter,
{
}

impl<T, Param> TraitParameterStorage<Param> for T
where
    T: Index<usize, Output = Param> + TraitParameterStorageUtils,
    Param: TraitParameter,
{
}

impl<T> TraitParameterStorageUtils for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

// 模型训练需要实现的接口
pub trait TraitParameterStorageTrainUtilsInit<Param>: Default {
    fn init(value: Param, size: usize) -> Self;
}
impl<Param: Num + Clone> TraitParameterStorageTrainUtilsInit<Param> for Vec<Param> {
    fn init(value: Param, size: usize) -> Self {
        vec![value; size]
    }
}
pub trait TraitParameterStorageTrainUtils<Param>:
    Clone
    + Index<usize, Output = Param>
    + IndexMut<usize, Output = Param>
    + TraitParameterStorageTrainUtilsInit<Param>
{
}
impl<T, Param> TraitParameterStorageTrainUtils<Param> for T where
    T: Clone
        + Index<usize, Output = Param>
        + IndexMut<usize, Output = Param>
        + TraitParameterStorageTrainUtilsInit<Param>
{
}

// 模型压缩需要实现的接口
pub trait TraitParameterStorageCompressUtils<Param> {
    fn with_capacity(capacity: usize) -> Self;
    fn push(&mut self, value: Param);
}

impl<T> TraitParameterStorageCompressUtils<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn push(&mut self, value: T) {
        self.push(value);
    }
}
