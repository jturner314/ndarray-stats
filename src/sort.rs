use indexmap::IndexMap;
use ndarray::prelude::*;
use ndarray::{Data, DataMut, Slice};

/// Methods for sorting and partitioning 1-D arrays.
pub trait Sort1dExt<A, S>
where
    S: Data<Elem = A>,
{
    /// Return the element that would occupy the `i`-th position if
    /// the array were sorted in increasing order.
    ///
    /// The array is shuffled **in place** to retrieve the desired element:
    /// no copy of the array is allocated.
    /// After the shuffling, all elements with an index smaller than `i`
    /// are smaller than the desired element, while all elements with
    /// an index greater or equal than `i` are greater than or equal
    /// to the desired element.
    ///
    /// No other assumptions should be made on the ordering of the
    /// elements after this computation.
    ///
    /// This method performs [Sesquickselect].
    ///
    /// [Sesquickselect]: https://www.wild-inter.net/publications/martinez-nebel-wild-2019.pdf
    ///
    /// **Panics** if `i` is greater than or equal to `n`.
    fn get_from_sorted_mut(&mut self, i: usize) -> A
    where
        A: Ord + Clone,
        S: DataMut;

    /// A bulk version of [`get_from_sorted_mut`], optimized to retrieve multiple
    /// indexes at once.
    /// It returns an `IndexMap`, with indexes as keys and retrieved elements as
    /// values.
    /// The `IndexMap` is sorted with respect to indexes in increasing order:
    /// this ordering is preserved when you iterate over it (using `iter`/`into_iter`).
    ///
    /// **Panics** if any element in `indexes` is greater than or equal to `n`,
    /// where `n` is the length of the array..
    ///
    /// [`get_from_sorted_mut`]: #tymethod.get_from_sorted_mut
    fn get_many_from_sorted_mut<S2>(&mut self, indexes: &ArrayBase<S2, Ix1>) -> IndexMap<usize, A>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = usize>;

    /// Partitions the array in increasing order based on the value initially
    /// located at `pivot_index` and returns the new index of the value.
    ///
    /// The elements are rearranged in such a way that the value initially
    /// located at `pivot_index` is moved to the position it would be in an
    /// array sorted in increasing order. The return value is the new index of
    /// the value after rearrangement. All elements smaller than the value are
    /// moved to its left and all elements equal or greater than the value are
    /// moved to its right. The ordering of the elements in the two partitions
    /// is undefined.
    ///
    /// `self` is shuffled **in place** to operate the desired partition:
    /// no copy of the array is allocated.
    ///
    /// The method uses Hoare's partition algorithm.
    /// Complexity: O(`n`), where `n` is the number of elements in the array.
    /// Average number of element swaps: n/6 - 1/3 (see
    /// [link](https://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto/11550))
    ///
    /// **Panics** if `pivot_index` is greater than or equal to `n`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::Sort1dExt;
    ///
    /// let mut data = array![3, 1, 4, 5, 2];
    /// let pivot_index = 2;
    /// let pivot_value = data[pivot_index];
    ///
    /// // Partition by the value located at `pivot_index`.
    /// let new_index = data.partition_mut(pivot_index);
    /// // The pivot value is now located at `new_index`.
    /// assert_eq!(data[new_index], pivot_value);
    /// // Elements less than that value are moved to the left.
    /// for i in 0..new_index {
    ///     assert!(data[i] < pivot_value);
    /// }
    /// // Elements greater than or equal to that value are moved to the right.
    /// for i in (new_index + 1)..data.len() {
    ///      assert!(data[i] >= pivot_value);
    /// }
    /// ```
    fn partition_mut(&mut self, pivot_index: usize) -> usize
    where
        A: Ord + Clone,
        S: DataMut;

    /// Partitions the array in increasing order based on the values initially located at the two
    /// pivot indexes `lower` and `upper` and returns the new indexes of their values.
    ///
    /// The elements are rearranged in such a way that the two pivot values are moved to the indexes
    /// they would be in an array sorted in increasing order. The return values are the new indexes
    /// of the values after rearrangement. All elements less than the values are moved to their left
    /// and all elements equal or greater than the values are moved to their right. The ordering of
    /// the elements in the three partitions is undefined.
    ///
    /// The array is shuffled **in place**, no copy of the array is allocated.
    ///
    /// This method performs [dual-pivot partitioning].
    ///
    /// [dual-pivot partitioning]: https://www.wild-inter.net/publications/wild-2018b.pdf
    ///
    /// **Panics** if `lower` or `upper` is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::Sort1dExt;
    ///
    /// let mut data = array![3, 1, 4, 5, 2];
    /// // Skewed pivot values.
    /// let (lower_value, upper_value) = (1, 5);
    ///
    /// // Partitions by the values located at `1` and `3`.
    /// let (lower_index, upper_index) = data.dual_partition_mut(1, 3);
    /// // The pivot values are now located at `lower_index` and `upper_index`.
    /// assert_eq!(data[lower_index], lower_value);
    /// assert_eq!(data[upper_index], upper_value);
    /// // Elements lower than the lower pivot value are moved to its left.
    /// for i in 0..lower_index {
    ///     assert!(data[i] < lower_value);
    /// }
    /// // Elements greater than or equal the lower pivot value and less than or equal the upper
    /// // pivot value are moved between the two pivot indexes.
    /// for i in lower_index + 1..upper_index {
    ///     assert!(lower_value <= data[i]);
    ///     assert!(data[i] <= upper_value);
    /// }
    /// // Elements greater than or equal the upper pivot value are moved to its right.
    /// for i in upper_index + 1..data.len() {
    ///     assert!(upper_value <= data[i]);
    /// }
    /// ```
    fn dual_partition_mut(&mut self, lower: usize, upper: usize) -> (usize, usize)
    where
        A: Ord + Clone,
        S: DataMut;

    private_decl! {}
}

impl<A, S> Sort1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn get_from_sorted_mut(&mut self, i: usize) -> A
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let n = self.len();
        // Recursion cutoff at integer multiple of sample space divider of 7 elements.
        if n < 21 {
            for mut index in 1..n {
                while index > 0 && self[index - 1] > self[index] {
                    self.swap(index - 1, index);
                    index -= 1;
                }
            }
            self[i].clone()
        } else {
            // Sorted sample of 5 equally spaced elements around the center.
            let mut sample = [0; 5];
            sample_mut(self, &mut sample);
            // Adapt pivot sampling to relative sought rank and switch from dual-pivot to
            // single-pivot partitioning for extreme sought ranks.
            let sought_rank = i as f64 / n as f64;
            if (0.036..=0.964).contains(&sought_rank) {
                let (lower_index, upper_index) = if sought_rank <= 0.5 {
                    if sought_rank <= 0.153 {
                        (0, 1) // (0, 0, 3)
                    } else {
                        (0, 2) // (0, 1, 2)
                    }
                } else {
                    if sought_rank <= 0.847 {
                        (2, 4) // (2, 1, 0)
                    } else {
                        (3, 4) // (3, 0, 0)
                    }
                };
                let (lower_index, upper_index) =
                    self.dual_partition_mut(sample[lower_index], sample[upper_index]);
                if i < lower_index {
                    self.slice_axis_mut(Axis(0), Slice::from(..lower_index))
                        .get_from_sorted_mut(i)
                } else if i == lower_index {
                    self[i].clone()
                } else if i < upper_index {
                    self.slice_axis_mut(Axis(0), Slice::from(lower_index + 1..upper_index))
                        .get_from_sorted_mut(i - (lower_index + 1))
                } else if i == upper_index {
                    self[i].clone()
                } else {
                    self.slice_axis_mut(Axis(0), Slice::from(upper_index + 1..))
                        .get_from_sorted_mut(i - (upper_index + 1))
                }
            } else {
                let pivot_index = if sought_rank <= 0.5 {
                    0 // (0, 4)
                } else {
                    4 // (4, 0)
                };
                let pivot_index = self.partition_mut(sample[pivot_index]);
                if i < pivot_index {
                    self.slice_axis_mut(Axis(0), Slice::from(..pivot_index))
                        .get_from_sorted_mut(i)
                } else if i == pivot_index {
                    self[i].clone()
                } else {
                    self.slice_axis_mut(Axis(0), Slice::from(pivot_index + 1..))
                        .get_from_sorted_mut(i - (pivot_index + 1))
                }
            }
        }
    }

    fn get_many_from_sorted_mut<S2>(&mut self, indexes: &ArrayBase<S2, Ix1>) -> IndexMap<usize, A>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = usize>,
    {
        let mut deduped_indexes: Vec<usize> = indexes.to_vec();
        deduped_indexes.sort_unstable();
        deduped_indexes.dedup();

        get_many_from_sorted_mut_unchecked(self, &deduped_indexes)
    }

    fn partition_mut(&mut self, pivot_index: usize) -> usize
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let pivot_value = self[pivot_index].clone();
        self.swap(pivot_index, 0);
        let n = self.len();
        let mut i = 1;
        let mut j = n - 1;
        loop {
            loop {
                if i > j {
                    break;
                }
                if self[i] >= pivot_value {
                    break;
                }
                i += 1;
            }
            while pivot_value <= self[j] {
                if j == 1 {
                    break;
                }
                j -= 1;
            }
            if i >= j {
                break;
            } else {
                self.swap(i, j);
                i += 1;
                j -= 1;
            }
        }
        self.swap(0, i - 1);
        i - 1
    }

    fn dual_partition_mut(&mut self, lower: usize, upper: usize) -> (usize, usize)
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let lowermost = 0;
        let uppermost = self.len() - 1;
        // Swap pivots with outermost elements.
        self.swap(lowermost, lower);
        self.swap(uppermost, upper);
        if self[lowermost] > self[uppermost] {
            // Sort pivots instead of panicking via assertion.
            self.swap(lowermost, uppermost);
        }
        // Increasing running and partition index starting after lower pivot.
        let mut index = lowermost + 1;
        let mut lower = lowermost + 1;
        // Decreasing partition index starting before upper pivot.
        let mut upper = uppermost - 1;
        // Swap elements at `index` into their partitions.
        while index <= upper {
            if self[index] < self[lowermost] {
                // Swap elements into lower partition.
                self.swap(index, lower);
                lower += 1;
            } else if self[index] >= self[uppermost] {
                // Search first element of upper partition.
                while self[upper] > self[uppermost] && index < upper {
                    upper -= 1;
                }
                // Swap elements into upper partition.
                self.swap(index, upper);
                if self[index] < self[lowermost] {
                    // Swap swapped elements into lower partition.
                    self.swap(index, lower);
                    lower += 1;
                }
                upper -= 1;
            }
            index += 1;
        }
        lower -= 1;
        upper += 1;
        // Swap pivots to their new indexes.
        self.swap(lowermost, lower);
        self.swap(uppermost, upper);
        // Lower and upper pivot index.
        (lower, upper)
    }

    private_impl! {}
}

/// To retrieve multiple indexes from the sorted array in an optimized fashion,
/// [get_many_from_sorted_mut] first of all sorts and deduplicates the
/// `indexes` vector.
///
/// `get_many_from_sorted_mut_unchecked` does not perform this sorting and
/// deduplication, assuming that the user has already taken care of it.
///
/// Useful when you have to call [get_many_from_sorted_mut] multiple times
/// using the same indexes.
///
/// [get_many_from_sorted_mut]: ../trait.Sort1dExt.html#tymethod.get_many_from_sorted_mut
pub(crate) fn get_many_from_sorted_mut_unchecked<A, S>(
    array: &mut ArrayBase<S, Ix1>,
    indexes: &[usize],
) -> IndexMap<usize, A>
where
    A: Ord + Clone,
    S: DataMut<Elem = A>,
{
    if indexes.is_empty() {
        return IndexMap::new();
    }

    // Since `!indexes.is_empty()` and indexes must be in-bounds, `array` must
    // be non-empty.
    let mut values = vec![array[0].clone(); indexes.len()];
    _get_many_from_sorted_mut_unchecked(array.view_mut(), &mut indexes.to_owned(), &mut values);

    // We convert the vector to a more search-friendly `IndexMap`.
    indexes.iter().cloned().zip(values.into_iter()).collect()
}

/// This is the recursive portion of `get_many_from_sorted_mut_unchecked`.
///
/// `indexes` is the list of indexes to get. `indexes` is mutable so that it
/// can be used as scratch space for this routine; the value of `indexes` after
/// calling this routine should be ignored.
///
/// `values` is a pre-allocated slice to use for writing the output. Its
/// initial element values are ignored.
fn _get_many_from_sorted_mut_unchecked<A>(
    mut array: ArrayViewMut1<'_, A>,
    indexes: &mut [usize],
    values: &mut [A],
) where
    A: Ord + Clone,
{
    let n = array.len();
    debug_assert!(n >= indexes.len()); // because indexes must be unique and in-bounds
    debug_assert_eq!(indexes.len(), values.len());

    if indexes.is_empty() {
        // Nothing to do in this case.
        return;
    }

    // Recursion cutoff at integer multiple of sample space divider of 7 elements.
    if n < 21 {
        for mut index in 1..n {
            while index > 0 && array[index - 1] > array[index] {
                array.swap(index - 1, index);
                index -= 1;
            }
        }
        for (value, index) in values.iter_mut().zip(indexes.iter()) {
            *value = array[*index].clone();
        }
        return;
    }

    // Sorted sample of 5 equally spaced elements around the center.
    let mut sample = [0; 5];
    sample_mut(&mut array, &mut sample);
    let (lower_index, upper_index) = if indexes.len() == 1 {
        // Adapt pivot sampling to relative sought rank and switch from dual-pivot to single-pivot
        // partitioning for extreme sought ranks.
        let sought_rank = indexes[0] as f64 / n as f64;
        if (0.036..=0.964).contains(&sought_rank) {
            if sought_rank <= 0.5 {
                if sought_rank <= 0.153 {
                    (0, 1) // (0, 0, 3)
                } else {
                    (0, 2) // (0, 1, 2)
                }
            } else {
                if sought_rank <= 0.847 {
                    (2, 4) // (2, 1, 0)
                } else {
                    (3, 4) // (3, 0, 0)
                }
            }
        } else {
            let pivot_index = if sought_rank <= 0.5 {
                0 // (0, 4)
            } else {
                4 // (4, 0)
            };

            // We partition the array with respect to the pivot value. The pivot value moves to the
            // new `pivot_index`.
            //
            // Elements strictly less than the pivot value have indexes < `pivot_index`.
            //
            // Elements greater than or equal the pivot value have indexes > `pivot_index`.
            let pivot_index = array.partition_mut(sample[pivot_index]);
            let (found_exact, split_index) = match indexes.binary_search(&pivot_index) {
                Ok(index) => (true, index),
                Err(index) => (false, index),
            };
            let (lower_indexes, upper_indexes) = indexes.split_at_mut(split_index);
            let (lower_values, upper_values) = values.split_at_mut(split_index);
            let (upper_indexes, upper_values) = if found_exact {
                upper_values[0] = array[pivot_index].clone(); // Write exactly found value.
                (&mut upper_indexes[1..], &mut upper_values[1..])
            } else {
                (upper_indexes, upper_values)
            };

            // We search recursively for the values corresponding to indexes strictly less than
            // `pivot_index` in the lower partition.
            _get_many_from_sorted_mut_unchecked(
                array.slice_axis_mut(Axis(0), Slice::from(..pivot_index)),
                lower_indexes,
                lower_values,
            );

            // We search recursively for the values corresponding to indexes greater than or equal
            // `pivot_index` in the upper partition. Since only the upper partition of the array is
            // passed in, the indexes need to be shifted by length of the lower partition.
            upper_indexes.iter_mut().for_each(|x| *x -= pivot_index + 1);
            _get_many_from_sorted_mut_unchecked(
                array.slice_axis_mut(Axis(0), Slice::from(pivot_index + 1..)),
                upper_indexes,
                upper_values,
            );

            return;
        }
    } else {
        // Since there is no single sought rank to adapt pivot sampling to, the recommended skewed
        // pivot sampling of dual-pivot Quicksort is used in the assumption that multiple indexes
        // change characteristics from Quickselect towards Quicksort.
        (0, 2) // (0, 1, 2)
    };

    // We partition the array with respect to the two pivot values. The pivot values move to the new
    // `lower_index` and `upper_index`.
    //
    // Elements strictly less than the lower pivot value have indexes < `lower_index`.
    //
    // Elements greater than or equal the lower pivot value and less than or equal the upper pivot
    // value have indexes > `lower_index` and < `upper_index`.
    //
    // Elements greater than or equal the upper pivot value have indexes > `upper_index`.
    let (lower_index, upper_index) =
        array.dual_partition_mut(sample[lower_index], sample[upper_index]);

    // We use a divide-and-conquer strategy, splitting the indexes we are searching for (`indexes`)
    // and the corresponding portions of the output slice (`values`) into partitions with respect to
    // `lower_index` and `upper_index`.
    let (found_exact, split_index) = match indexes.binary_search(&lower_index) {
        Ok(index) => (true, index),
        Err(index) => (false, index),
    };
    let (lower_indexes, inner_indexes) = indexes.split_at_mut(split_index);
    let (lower_values, inner_values) = values.split_at_mut(split_index);
    let (upper_indexes, upper_values) = if found_exact {
        inner_values[0] = array[lower_index].clone(); // Write exactly found value.
        (&mut inner_indexes[1..], &mut inner_values[1..])
    } else {
        (inner_indexes, inner_values)
    };

    let (found_exact, split_index) = match upper_indexes.binary_search(&upper_index) {
        Ok(index) => (true, index),
        Err(index) => (false, index),
    };
    let (inner_indexes, upper_indexes) = upper_indexes.split_at_mut(split_index);
    let (inner_values, upper_values) = upper_values.split_at_mut(split_index);
    let (upper_indexes, upper_values) = if found_exact {
        upper_values[0] = array[upper_index].clone(); // Write exactly found value.
        (&mut upper_indexes[1..], &mut upper_values[1..])
    } else {
        (upper_indexes, upper_values)
    };

    // We search recursively for the values corresponding to indexes strictly less than
    // `lower_index` in the lower partition.
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(..lower_index)),
        lower_indexes,
        lower_values,
    );

    // We search recursively for the values corresponding to indexes greater than or equal
    // `lower_index` in the inner partition, that is between the lower and upper partition. Since
    // only the inner partition of the array is passed in, the indexes need to be shifted by length
    // of the lower partition.
    inner_indexes.iter_mut().for_each(|x| *x -= lower_index + 1);
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(lower_index + 1..upper_index)),
        inner_indexes,
        inner_values,
    );

    // We search recursively for the values corresponding to indexes greater than or equal
    // `upper_index` in the upper partition. Since only the upper partition of the array is passed
    // in, the indexes need to be shifted by the combined length of the lower and inner partition.
    upper_indexes.iter_mut().for_each(|x| *x -= upper_index + 1);
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(upper_index + 1..)),
        upper_indexes,
        upper_values,
    );
}

/// Equally space `sample` indexes around the center of `array` and sort them by their values.
///
/// `sample` content is ignored but its length defines the sample size and the sample space divider.
///
/// Assumes arrays of at least `sample.len() + 2` elements.
pub(crate) fn sample_mut<A, S>(array: &mut ArrayBase<S, Ix1>, sample: &mut [usize])
where
    A: Ord + Clone,
    S: DataMut<Elem = A>,
{
    // Space between sample indexes.
    let space = array.len() / (sample.len() + 2);
    // Lowermost sample index.
    let lowermost = array.len() / 2 - (sample.len() / 2) * space;
    // Equally space sample indexes and sort them by their values by looking up their indexes.
    for mut index in 1..sample.len() {
        // Equally space sample indexes based on their lowermost index.
        sample[index] = lowermost + index * space;
        // Insertion sort looking up only the already equally spaced sample indexes.
        while index > 0 && array[sample[index - 1]] > array[sample[index]] {
            array.swap(sample[index - 1], sample[index]);
            index -= 1;
        }
    }
}
