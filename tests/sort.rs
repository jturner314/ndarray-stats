use ndarray::prelude::*;
use ndarray_stats::Sort1dExt;
use quickcheck_macros::quickcheck;

#[test]
fn test_partition_mut() {
    let mut l = vec![
        arr1(&[1, 1, 1, 1, 1]),
        arr1(&[1, 3, 2, 10, 10]),
        arr1(&[2, 3, 4, 1]),
        arr1(&[
            355, 453, 452, 391, 289, 343, 44, 154, 271, 44, 314, 276, 160, 469, 191, 138, 163, 308,
            395, 3, 416, 391, 210, 354, 200,
        ]),
        arr1(&[
            84, 192, 216, 159, 89, 296, 35, 213, 456, 278, 98, 52, 308, 418, 329, 173, 286, 106,
            366, 129, 125, 450, 23, 463, 151,
        ]),
    ];
    for a in l.iter_mut() {
        let n = a.len();
        let pivot_index = n - 1;
        let pivot_value = a[pivot_index].clone();
        let partition_index = a.partition_mut(pivot_index);
        for i in 0..partition_index {
            assert!(a[i] < pivot_value);
        }
        assert_eq!(a[partition_index], pivot_value);
        for j in (partition_index + 1)..n {
            assert!(pivot_value <= a[j]);
        }
    }
}

#[test]
fn test_dual_partition_mut() {
    let mut l = vec![
        arr1(&[1, 1, 1, 1, 1]),
        arr1(&[1, 3, 2, 10, 10]),
        arr1(&[2, 3, 4, 1]),
        arr1(&[
            355, 453, 452, 391, 289, 343, 44, 154, 271, 44, 314, 276, 160, 469, 191, 138, 163, 308,
            395, 3, 416, 391, 210, 354, 200,
        ]),
        arr1(&[
            84, 192, 216, 159, 89, 296, 35, 213, 456, 278, 98, 52, 308, 418, 329, 173, 286, 106,
            366, 129, 125, 450, 23, 463, 151,
        ]),
    ];
    for a in l.iter_mut() {
        let n = a.len();
        let (lower_index, upper_index) = a.dual_partition_mut(1, a.len() / 2);
        for i in 0..lower_index {
            assert!(a[i] < a[lower_index]);
        }
        for i in lower_index + 1..upper_index {
            assert!(a[lower_index] <= a[i]);
            assert!(a[i] <= a[upper_index]);
        }
        for i in (upper_index + 1)..n {
            assert!(a[upper_index] <= a[i]);
        }
    }
}

#[test]
fn test_sorted_get_mut() {
    let a = arr1(&[1, 3, 2, 10]);
    let j = a.clone().view_mut().get_from_sorted_mut(2);
    assert_eq!(j, 3);
    let j = a.clone().view_mut().get_from_sorted_mut(1);
    assert_eq!(j, 2);
    let j = a.clone().view_mut().get_from_sorted_mut(3);
    assert_eq!(j, 10);
}

#[quickcheck]
fn test_sorted_get_many_mut(mut xs: Vec<i64>) -> bool {
    let n = xs.len();
    if n == 0 {
        true
    } else {
        let mut v = Array::from(xs.clone());

        // Insert each index twice, to get a set of indexes with duplicates, not sorted
        let mut indexes: Vec<usize> = (0..n).into_iter().collect();
        indexes.append(&mut (0..n).collect());

        let mut sorted_v = Vec::with_capacity(n);
        for (i, (key, value)) in v
            .get_many_from_sorted_mut(&Array::from(indexes))
            .into_iter()
            .enumerate()
        {
            if i != key {
                return false;
            }
            sorted_v.push(value);
        }
        xs.sort();
        println!("Sorted: {:?}. Truth: {:?}", sorted_v, xs);
        xs == sorted_v
    }
}

#[quickcheck]
fn test_sorted_get_mut_as_sorting_algorithm(mut xs: Vec<i64>) -> bool {
    let n = xs.len();
    if n == 0 {
        true
    } else {
        let mut v = Array::from(xs.clone());
        let sorted_v: Vec<_> = (0..n).map(|i| v.get_from_sorted_mut(i)).collect();
        xs.sort();
        xs == sorted_v
    }
}
