use ndarray::prelude::*;
use ndarray::Data;
use super::bins::Bins;
use super::errors::BinNotFound;

pub struct HistogramCounts<A: Ord> {
    counts: ArrayD<usize>,
    bins: Vec<Bins<A>>,
}

impl<A: Ord> HistogramCounts<A> {
    pub fn new(edges: Vec<Bins<A>>) -> Self {
        let counts = ArrayD::zeros(
            edges.iter().map(|e| e.len()
            ).collect::<Vec<_>>());
        HistogramCounts { counts, bins: edges }
    }

    pub fn add_observation(&mut self, observation: ArrayView1<A>) -> Result<(), BinNotFound> {
        let bin = observation
            .iter()
            .zip(&self.bins)
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()?;
        self.counts[IxDyn(&bin)] += 1;
        Ok(())
    }
}

/// Histogram methods.
pub trait HistogramExt<A, S>
    where
        S: Data<Elem = A>,
{

    fn histogram(&self, bins: Vec<Bins<A>>) -> HistogramCounts<A>
        where
            A: Ord;
}