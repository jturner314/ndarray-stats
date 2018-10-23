use ndarray::prelude::*;
use ndarray::Data;
use super::bins::Bins;
use super::errors::BinNotFound;

/// Histogram data structure.
pub struct Histogram<A: Ord> {
    counts: ArrayD<usize>,
    bins: Vec<Bins<A>>,
}

impl<A: Ord> Histogram<A> {
    /// Return a new instance of Histogram given
    /// a vector of [`Bins`].
    ///
    /// The `i`-th element in `Vec<Bins<A>>` represents the 1-dimensional
    /// projection of the bin grid on the `i`-th axis.
    ///
    /// [`Bins`]: struct.Bins.html
    pub fn new(bins: Vec<Bins<A>>) -> Self {
        let counts = ArrayD::zeros(
            bins.iter().map(|e| e.len()
            ).collect::<Vec<_>>());
        Histogram { counts, bins }
    }

    /// Add a single observation to the histogram.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != observation.len()`.
    ///
    /// # Example:
    /// ```
    /// extern crate ndarray_stats;
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate noisy_float;
    /// use ndarray_stats::histogram::{Edges, Bins, Histogram};
    /// use noisy_float::types::n64;
    ///
    /// # fn main() {
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = vec![bins.clone(), bins.clone()];
    /// let mut histogram = Histogram::new(square_grid);
    ///
    /// let observation = array![n64(0.5), n64(0.6)];
    ///
    /// histogram.add_observation(observation.view());
    ///
    /// let histogram_matrix = histogram.as_view();
    /// assert_eq!(histogram_matrix[[1, 1]], 1);
    /// # }
    /// ```
    pub fn add_observation(&mut self, observation: ArrayView1<A>) -> Result<(), BinNotFound> {
        assert_eq!(
            self.ndim(),
            observation.len(),
            "Dimensions do not match: observation has {0} dimensions, \
             while the histogram has {1}.", observation.len(), self.ndim()
        );
        let bin = observation
            .iter()
            .zip(&self.bins)
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()?;
        self.counts[IxDyn(&bin)] += 1;
        Ok(())
    }

    /// Returns the number of dimensions of the space the histogram is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.counts.ndim(), self.bins.len());
        self.counts.ndim()
    }

    /// Borrow a view to the histogram matrix.
    pub fn as_view(&self) -> ArrayViewD<usize> {
        self.counts.view()
    }
}

/// Extension trait for `ArrayBase` providing methods to compute histograms.
pub trait HistogramExt<A, S>
    where
        S: Data<Elem = A>,
{
    /// Return the [histogram](https://en.wikipedia.org/wiki/Histogram)
    /// for a 2-dimensional array of points `M`.
    ///
    /// Let `(n, d)` be the shape of `M`:
    /// - `n` is the number of points;
    /// - `d` is the number of dimensions of the space those points belong to.
    /// It follows that every column in `M` is a `d`-dimensional point.
    ///
    /// For example: a (3, 4) matrix `M` is a collection of 3 points in a
    /// 4-dimensional space.
    ///
    /// **Panics** if `d` is different from `bins.len()`.
    fn histogram(&self, bins: Vec<Bins<A>>) -> Histogram<A>
        where
            A: Ord;
}

impl<A, S> HistogramExt<A, S> for ArrayBase<S, Ix2>
    where
        S: Data<Elem = A>,
        A: Ord,
{
    fn histogram(&self, bins: Vec<Bins<A>>) -> Histogram<A>
    {
        let mut histogram = Histogram::new(bins);
        for point in self.axis_iter(Axis(0)) {
            histogram.add_observation(point);
        }
        histogram
    }
}
