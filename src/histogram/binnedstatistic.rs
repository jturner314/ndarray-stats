use super::errors::BinNotFound;
use super::grid::Grid;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use ndarray::prelude::{array, ArrayBase, ArrayD, ArrayViewD, Axis, Ix1, Ix2};
use ndarray::{Data, Zip};
use num_traits::{
    identities::{One, Zero},
    Float,
};

/// Binned statistic data structure.
#[derive(Clone, Debug)]
pub struct BinnedStatistic<A: Ord, T: Zero> {
    count: ArrayD<usize>,
    number: ArrayD<T>,
    sum: ArrayD<T>,
    mean: ArrayD<T>,
    variance: ArrayD<T>,
    standard_deviation: ArrayD<T>,
    min: ArrayD<T>,
    max: ArrayD<T>,
    grid: Grid<A>,
}

impl<A, T> BinnedStatistic<A, T>
where
    A: Ord,
    T: Float,
{
    /// Returns a new instance of BinnedStatistic given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>) -> Self {
        let count = ArrayD::zeros(grid.shape());
        let number = ArrayD::zeros(grid.shape());
        let sum = ArrayD::zeros(grid.shape());
        let mean = ArrayD::zeros(grid.shape());
        let variance = ArrayD::zeros(grid.shape());
        let standard_deviation = ArrayD::zeros(grid.shape());
        let min = ArrayD::from_elem(grid.shape(), T::infinity());
        let max = ArrayD::from_elem(grid.shape(), T::neg_infinity());
        BinnedStatistic {
            count,
            number,
            sum,
            mean,
            variance,
            standard_deviation,
            min,
            max,
            grid,
        }
    }

    /// Adds a single sample to the binned statistic.
    ///
    /// Possible binned statistics are:
    /// * `count`: (equivalent to histogram).  
    /// * `number`: (equivalent to histogram but different data type).
    /// * `sum` computes the sum of values for points within each bin. This is identical to
    /// a weighted histogram.
    /// * `mean`: computes the mean of values for points within each bin. Empty bins will be
    /// represented by zero.
    /// * `variance`: computes the variance of values for points within each bin. Empty bins will be
    /// represented by zero.
    /// * `standard_deviation`: computes the standard deviation of values for points within each bin.
    /// Empty bins will be represented by zero.
    /// * `min`: computes the minimum of values for points within each bin. Empty bins will be
    /// represented by `inf`.
    /// * `max`: computes the maximum of values for points within each bin. Empty bins will be
    /// represented by `-inf`.
    ///
    /// Alternatively arrays of `BinContent`s can be computed indicating empty bins with `Empty`
    /// and filled bins with `Value(x)`.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != sample.len()`.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_sum = binned_statistic.sum();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 3.0],
    /// ];
    /// assert_eq!(binned_statistic_sum, expected.into_dyn());
    ///
    /// let binned_statistic_bc = binned_statistic.sum_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(3.0))],
    /// ];
    /// assert_eq!(binned_statistic_bc, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn add_sample<S>(&mut self, sample: &ArrayBase<S, Ix1>, value: T) -> Result<(), BinNotFound>
    where
        S: Data<Elem = A>,
        T: Float,
    {
        match self.grid.index_of(sample) {
            Some(bin_index) => {
                let id = &*bin_index;

                // Saving count
                let n1 = self.number[id];

                // Calculate count & sum
                self.count[id] = self.count[id] + 1usize;
                self.number[id] = self.number[id] + T::one();
                self.sum[id] = self.sum[id] + value;

                // Mean & variance
                let n = self.number[id];
                let delta = value - self.mean[id];
                let delta_n = delta / n;
                let term1 = delta * delta_n * n1;

                self.mean[id] = self.mean[id] + delta_n;
                self.variance[id] = (self.variance[id] * n1 + term1) / n;
                self.standard_deviation[id] = self.variance[id].sqrt();

                // Min & max
                self.min[id] = Float::min(self.min[id], value);
                self.max[id] = Float::max(self.max[id], value);

                Ok(())
            }
            None => Err(BinNotFound),
        }
    }

    /// Returns the number of dimensions of the space the binned statistic is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.count.ndim(), self.grid.ndim());
        self.count.ndim()
    }

    /// Borrows a view on the binned statistic `count` matrix (equivalent to histogram).
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_count = binned_statistic.count();
    /// let expected = array![
    ///     [0, 0],
    ///     [0, 2],
    /// ];
    /// assert_eq!(binned_statistic_count, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn count(&self) -> ArrayViewD<'_, usize> {
        self.count.view()
    }

    /// Borrows a view on the binned statistic `number` matrix (equivalent to histogram).
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_number = binned_statistic.number();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 2.0],
    /// ];
    /// assert_eq!(binned_statistic_number, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn number(&self) -> ArrayViewD<'_, T> {
        self.number.view()
    }

    /// Borrows a view on the binned statistic `sum` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_sum = binned_statistic.sum();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 3.0],
    /// ];
    /// assert_eq!(binned_statistic_sum, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn sum(&self) -> ArrayViewD<'_, T> {
        self.sum.view()
    }

    /// Borrows a view on the binned statistic `mean` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_mean = binned_statistic.mean();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 1.5],
    /// ];
    /// assert_eq!(binned_statistic_mean, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn mean(&self) -> ArrayViewD<'_, T> {
        self.mean.view()
    }

    /// Borrows a view on the binned statistic `variance` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_variance = binned_statistic.variance();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 0.25],
    /// ];
    /// assert_eq!(binned_statistic_variance, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn variance(&self) -> ArrayViewD<'_, T> {
        self.variance.view()
    }

    /// Borrows a view on the binned statistic `standard_deviation` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_standard_deviation = binned_statistic.standard_deviation();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 0.5],
    /// ];
    /// assert_eq!(binned_statistic_standard_deviation, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn standard_deviation(&self) -> ArrayViewD<'_, T> {
        self.standard_deviation.view()
    }

    /// Borrows a view on the binned statistic `min` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    /// use num_traits::Float;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_min = binned_statistic.min();
    /// let expected = array![
    ///     [f64::infinity(), f64::infinity()],
    ///     [f64::infinity(), 1.0],
    /// ];
    /// assert_eq!(binned_statistic_min, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn min(&self) -> ArrayViewD<'_, T> {
        self.min.view()
    }

    /// Borrows a view on the binned statistic `max` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    /// use num_traits::Float;
    ///
    /// let bins = Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.)]));
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_max = binned_statistic.max();
    /// let expected = array![
    ///     [f64::neg_infinity(), f64::neg_infinity()],
    ///     [f64::neg_infinity(), 2.0],
    /// ];
    /// assert_eq!(binned_statistic_max, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn max(&self) -> ArrayViewD<'_, T> {
        self.max.view()
    }

    /// Borrows an immutable reference to the binned statistic grid.
    pub fn grid(&self) -> &Grid<A> {
        &self.grid
    }

    /// Returns an array of `BinContent`s of the `count` matrix (equivalent to histogram).
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_count = binned_statistic.count_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(2)],
    /// ];
    /// assert_eq!(binned_statistic_count, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn count_binned(&self) -> ArrayD<BinContent<usize>> {
        let mut count_binned = ArrayD::<BinContent<usize>>::zeros(self.count.shape());

        for (count_arr, binned) in self.count.iter().zip(&mut count_binned) {
            *binned = if *count_arr == 0usize {
                BinContent::Empty
            } else {
                BinContent::Value(*count_arr)
            };
        }
        count_binned
    }

    /// Returns an array of `BinContents`s of the `number` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_number = binned_statistic.number_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(2.0))],
    /// ];
    /// assert_eq!(binned_statistic_number, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn number_binned(&self) -> ArrayD<BinContent<T>> {
        let mut number_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut number_binned)
            .and(&self.number)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        number_binned
    }

    /// Returns an array of `BinContents`s of the `sum` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_sum = binned_statistic.sum_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(3.0))],
    /// ];
    /// assert_eq!(binned_statistic_sum, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn sum_binned(&self) -> ArrayD<BinContent<T>> {
        let mut sum_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut sum_binned)
            .and(&self.sum)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        sum_binned
    }

    /// Returns an array of `BinContents`s of the `mean` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_mean = binned_statistic.mean_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(1.5))],
    /// ];
    /// assert_eq!(binned_statistic_mean, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn mean_binned(&self) -> ArrayD<BinContent<T>> {
        let mut mean_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut mean_binned)
            .and(&self.mean)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        mean_binned
    }

    /// Returns an array of `BinContents`s of the `variance` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_variance = binned_statistic.variance_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(0.25))],
    /// ];
    /// assert_eq!(binned_statistic_variance, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn variance_binned(&self) -> ArrayD<BinContent<T>> {
        let mut variance_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut variance_binned)
            .and(&self.variance)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        variance_binned
    }

    /// Returns an array of `BinContents`s of the `standard_deviation` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_standard_deviation = binned_statistic.standard_deviation_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(0.5))],
    /// ];
    /// assert_eq!(binned_statistic_standard_deviation, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn standard_deviation_binned(&self) -> ArrayD<BinContent<T>> {
        let mut standard_deviation_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut standard_deviation_binned)
            .and(&self.standard_deviation)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        standard_deviation_binned
    }

    /// Returns an array of `BinContents`s of the `min` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_min = binned_statistic.min_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(1.0))],
    /// ];
    /// assert_eq!(binned_statistic_min, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn min_binned(&self) -> ArrayD<BinContent<T>> {
        let mut min_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut min_binned)
            .and(&self.min)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        min_binned
    }

    /// Returns an array of `BinContents`s of the `max` matrix.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_mxa = binned_statistic.max_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(2.0))],
    /// ];
    /// assert_eq!(binned_statistic_mxa, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn max_binned(&self) -> ArrayD<BinContent<T>> {
        let mut max_binned = ArrayD::<BinContent<T>>::zeros(self.count.shape());

        Zip::from(&mut max_binned)
            .and(&self.max)
            .and(&self.count)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        max_binned
    }
}

// impl<A: Ord, T: Copy + num_traits::Num + Add<Output = T>> Add for BinnedStatistic<A, T> {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         if self.grid != other.grid {
//             panic!("`BinnedStatistics` can only be added for the same `grid`!")
//         };

//         BinnedStatistic {
//             count: &self.count + &other.count,
//             sum: &self.sum + &other.sum,
//             grid: self.grid,
//         }
//     }
// }

/// Extension trait for `ArrayBase` providing methods to compute binned statistics.
pub trait BinnedStatisticExt<A, S, T>
where
    S: Data<Elem = A>,
    T: Copy + Zero,
{
    /// Returns the binned statistic for a 1- or 2-dimensional array of samples `M`
    /// and a 1-dimensional vector of values `N`.
    ///
    /// Let `(n)` or `(n, d)` be the shape of `M` and `(n)` the shape of `N`:
    /// - `n` is the number of samples/values;
    /// - `d` is the number of dimensions of the space those points belong to.
    /// It follows that every column in `M` is a `d`-dimensional sample
    /// and every value in `N` is the corresponding value.
    ///
    /// For example: a (3, 4) matrix `M` is a collection of 3 points in a
    /// 4-dimensional space with a corresponding (4) vector `N`.
    ///
    /// Important: points outside the grid are ignored!
    ///
    /// **Panics** if `d` is different from `grid.ndim()`.
    ///
    /// # Example:
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::{
    ///     BinnedStatisticExt,
    ///     histogram::{BinnedStatistic, Grid, Edges, Bins},
    /// };
    /// use noisy_float::types::{N64, n64};
    ///
    /// let samples = array![
    ///     [n64(1.5), n64(0.5)],
    ///     [n64(-0.5), n64(1.5)],
    ///     [n64(-1.), n64(-0.5)],
    ///     [n64(0.5), n64(-1.)]
    /// ];
    /// let values = array![n64(12.), n64(-0.5), n64(1.), n64(2.)].into_dyn();
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.), n64(2.)]);
    /// let bins = Bins::new(edges);
    /// let grid = Grid::from(vec![bins.clone(), bins.clone()]);
    ///
    /// let binned_statistic = samples.binned_statistic(grid, values);
    ///
    /// // Bins are left inclusive, right exclusive!
    /// let expected_count = array![
    ///     [1, 0, 1],
    ///     [1, 0, 0],
    ///     [0, 1, 0]
    /// ];
    /// let expected_sum = array![
    ///     [n64(1.),  n64(0.),  n64(-0.5)],
    ///     [n64(2.),  n64(0.),  n64(0.)],
    ///     [n64(0.), n64(12.), n64(0.)]
    /// ];
    /// assert_eq!(binned_statistic.count(), expected_count.into_dyn());
    /// assert_eq!(binned_statistic.sum(), expected_sum.into_dyn());
    /// ```
    fn binned_statistic(&self, grid: Grid<A>, values: ArrayD<T>) -> BinnedStatistic<A, T>
    where
        A: Ord;

    private_decl! {}
}

/// Implementation of `BinnedStatisticExt` for `ArrayBase<S, Ix2>`.
impl<A, S, T> BinnedStatisticExt<A, S, T> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Ord,
    T: Float,
{
    fn binned_statistic(&self, grid: Grid<A>, values: ArrayD<T>) -> BinnedStatistic<A, T> {
        let mut binned_statistic = BinnedStatistic::new(grid);
        for (sample, value) in self.axis_iter(Axis(0)).zip(&values) {
            let _ = binned_statistic.add_sample(&sample, *value);
        }
        binned_statistic
    }

    private_impl! {}
}

/// Implementation of `BinnedStatisticExt` for `ArrayBase<S, Ix1>`.
impl<A, S, T> BinnedStatisticExt<A, S, T> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: Ord + Copy,
    T: Float,
{
    fn binned_statistic(&self, grid: Grid<A>, values: ArrayD<T>) -> BinnedStatistic<A, T> {
        let mut binned_statistic = BinnedStatistic::new(grid);
        for (sample, value) in self.iter().zip(&values) {
            let s = array![*sample];
            let _ = binned_statistic.add_sample(&s, *value);
        }
        binned_statistic
    }

    private_impl! {}
}

/// Indicator for empty fields or values for binned statistic
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinContent<T> {
    /// Empty bin
    Empty,
    /// Non-empty bin with some value `T`
    Value(T),
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T> BinContent<T> {
    /// Returns `true` if the bin contains a [`Value`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let x: BinContent<u32> = Value(2);
    /// assert_eq!(x.is_value(), true);
    ///
    /// let x: BinContent<u32> = Empty;
    /// assert_eq!(x.is_value(), false);
    /// ```
    pub fn is_value(&self) -> bool {
        match *self {
            Self::Value(_) => true,
            Self::Empty => false,
        }
    }

    /// Returns `true` if the BinContent is [`Empty`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let x: BinContent<u32> = Value(2);
    /// assert_eq!(x.is_empty(), false);
    ///
    /// let x: BinContent<u32> = Empty;
    /// assert_eq!(x.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        !self.is_value()
    }

    /// Returns `true` if the BinContent is a [`Value`] containing the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let x: BinContent<u32> = Value(2);
    /// assert_eq!(x.contains(&2), true);
    ///
    /// let x: BinContent<u32> = Value(3);
    /// assert_eq!(x.contains(&2), false);
    ///
    /// let x: BinContent<u32> = Empty;
    /// assert_eq!(x.contains(&2), false);
    /// ```
    pub fn contains<U>(&self, x: &U) -> bool
    where
        U: PartialEq<T>,
    {
        match self {
            Self::Value(y) => x == y,
            Self::Empty => false,
        }
    }

    /// Moves the value `v` out of the `BinContent<T>` if it is [`Value(v)`].
    ///
    /// In general, because this function may panic, its use is discouraged.
    /// Instead, prefer to use pattern matching and handle the [`Empty`]
    /// case explicitly.
    ///
    /// # Panics
    ///
    /// Panics if the self value equals [`Empty`].
    ///
    /// [`Value(v)`]: #variant.Value
    /// [`Empty`]: #variant.Empty
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let x: BinContent<u32> = Value(2);
    /// assert_eq!(x.unwrap(), 2);
    /// ```
    ///
    /// ```{.should_panic}
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let x: BinContent<u32> = Empty;
    /// assert_eq!(x.unwrap(), 2); // fails
    /// ```
    pub fn unwrap(self) -> T {
        match self {
            Self::Value(val) => val,
            Self::Empty => panic!("called `BinContent::unwrap()` on a `Empty` value"),
        }
    }

    /// Returns the contained value or a default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`unwrap_or_else`]: #method.unwrap_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// assert_eq!(Value(2).unwrap_or(5), 2);
    /// assert_eq!(Empty.unwrap_or(2), 2);
    /// ```
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Self::Value(x) => x,
            Self::Empty => default,
        }
    }

    /// Returns the contained value or computes it from a closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{BinContent, BinContent::Value, BinContent::Empty};
    ///
    /// let k = 10;
    /// assert_eq!(Value(4).unwrap_or_else(|| 2 * k), 4);
    /// assert_eq!(Empty.unwrap_or_else(|| 2 * k), 20);
    /// ```
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            Self::Value(x) => x,
            Self::Empty => f(),
        }
    }
}

/// Implementation of negation operator for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// assert_eq!(-bin, BinContent::Value(-2.0));
/// ```
impl<T: Neg<Output = T>> Neg for BinContent<T> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            BinContent::Empty => Self::Empty,
            BinContent::Value(v) => Self::Value(-v),
        }
    }
}

/// Implementation of addition for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin + bin, BinContent::<f64>::Value(4.0));
/// assert_eq!(bin + empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Add<Output = T>> Add for BinContent<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v + w),
        }
    }
}

/// Implementation of addition assignment  for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Add<Output = T> + Copy> AddAssign for BinContent<T> {
    fn add_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v + *w),
        };
    }
}

/// Implementation of substraction for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin - bin, BinContent::Value(0.0));
/// assert_eq!(bin - empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin - bin, BinContent::Value(-2.0));
/// assert_eq!(empty_bin - empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Neg<Output = T> + Sub<Output = T>> Sub for BinContent<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(-w),
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v - w),
        }
    }
}

/// Implementation of substraction assignment  for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Neg<Output = T> + Sub<Output = T> + Copy> SubAssign for BinContent<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(-w),
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v - *w),
        };
    }
}

/// Implementation of multiplication for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin * bin, BinContent::Value(4.0));
/// assert_eq!(bin * empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Mul<Output = T>> Mul for BinContent<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v * w),
        }
    }
}

/// Implementation of multiplication assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin *= bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// bin *= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin *= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin *= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Mul<Output = T> + Copy> MulAssign for BinContent<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v * *w),
        }
    }
}

/// Implementation of division for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin / bin, BinContent::Value(1.0));
/// assert_eq!(bin / empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Div<Output = T>> Div for BinContent<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v / w),
        }
    }
}

/// Implementation of division assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin /= bin;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin /= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin /= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin /= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Div<Output = T> + Copy> DivAssign for BinContent<T> {
    fn div_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v / *w),
        }
    }
}

/// Implementation of remainder for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(3.0);
/// let den = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin % den, BinContent::Value(1.0));
/// assert_eq!(bin % empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Rem<Output = T>> Rem for BinContent<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v % w),
        }
    }
}

/// Implementation of remainder assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(3.0);
/// let mut den = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin %= den;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin %= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(3.0);
/// empty_bin %= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin %= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: Rem<Output = T> + Copy> RemAssign for BinContent<T> {
    fn rem_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v % *w),
        }
    }
}

/// Implementation of zero-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::Zero;
///
/// let bin = BinContent::zero();
/// assert_eq!(bin, BinContent::<f64>::Empty);
/// ```
impl<T: Add<Output = T>> Zero for BinContent<T> {
    fn zero() -> Self {
        Self::Empty
    }
    fn is_zero(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Value(_) => false,
        }
    }
}

/// Implementation of one-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::One;
///
/// let bin = BinContent::one();
/// assert_eq!(bin, BinContent::Value(1.0));
/// ```
impl<T: num_traits::identities::One + PartialEq> One for BinContent<T> {
    fn one() -> Self {
        Self::Value(num_traits::identities::one())
    }
    fn is_one(&self) -> bool {
        match self {
            Self::Empty => false,
            Self::Value(v) => *v == num_traits::identities::one(),
        }
    }
}
