use std::mem::MaybeUninit;
use std::ops::Deref;

use crate::{
    board::{PencilMarks, RawCombinationsIter},
    generated_lut::COMBINATIONS,
};

// L: Lower bound for subsets length
// U: Upper bound for subsets length
// N: Size of set
#[derive(Debug)]
pub struct SubSetCache<const L: usize, const U: usize, const N: usize, T>
where
    [(); table_size(U, N)]:,
{
    table: [MaybeUninit<T>; table_size(U, N)],
}

impl<const N: usize, const L: usize, const U: usize, T> SubSetCache<L, U, N, T>
where
    [(); table_size(U, N)]:,
{
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: Fn(usize) -> T,
    {
        let mut table = MaybeUninit::uninit_array();
        for i in L..=U {
            let all_set = Self::combination_all_set();
            for index in RawCombinationsIter::new(all_set, i as u32) {
                table[index].write(cb(index));
            }
        }
        Self { table }
    }

    pub fn get(&self, key: usize) -> Option<&T> {
        if Self::key_is_valid(key) {
            let value = unsafe { &*self.table[key].as_ptr() };
            return Some(value);
        }
        None
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        if Self::key_is_valid(key) {
            let value = unsafe { &mut *self.table[key].as_mut_ptr() };
            return Some(value);
        }
        None
    }

    pub unsafe fn get_unchecked<'a>(&'a self, key: usize) -> &'a T {
        debug_assert!(Self::key_is_valid(key));

        unsafe { &*self.table.get_unchecked(key).as_ptr() }
    }

    pub unsafe fn get_unchecked_mut<'a>(&'a mut self, key: usize) -> &'a mut T {
        debug_assert!(Self::key_is_valid(key));

        unsafe { &mut *self.table.get_unchecked_mut(key).as_mut_ptr() }
    }

    pub fn entries(&self) -> EntriesIter {
        todo!()
    }

    pub fn entries_mut<'s>(&'s mut self) -> EntriesMutLUTIter<'s, L, U, N, T> {
        // if 2 <= L && 4 <= U {}
        assert_eq!(L, U, "subset ranges are currently not supported");
        // let all_set = Self::combination_all_set();
        EntriesMutLUTIter {
            it: COMBINATIONS[L - 2].into_iter(),
            ss_cache: self,
        }
    }

    // pub const fn entries_mut<Iter>(&mut self) -> EntriesMutIter<'_, L, U, N, T, E, Iter>
    // where
    //     Iter: Iterator<Item = impl Deref<Target = usize>,
    // {
    //     use crate::generated_lut::*;

    //     let k_cursor = L as u32;
    //     if 2 <= N && N <= 4 {
    //         EntriesMutIter {
    //             k_cursor,
    //             it: COMBINATIONS[L].into_iter(),
    //             ss_cache: self,
    //         }
    //     } else {
    //         let all_set = Self::combination_all_set();
    //         EntriesMutIter {
    //             k_cursor: L as u32,
    //             it: RawCombinationsIter::new(all_set, L as u32),
    //             ss_cache: self,
    //         }
    //     }
    // }

    pub fn entries_exact(&self, k: u32) -> EntriesExactIter<L, U, N, T> {
        let all_set = Self::combination_all_set();
        EntriesExactIter {
            it: RawCombinationsIter::new(all_set, k),
            ss_cache: self,
        }
    }

    pub fn entries_exact_mut(&mut self, k: usize) -> EntriesExactIterMut<L, U, N, T> {
        EntriesExactIterMut {
            k,
            cursor: 0,
            ss_cache: self,
        }
    }

    const fn key_is_valid(key: usize) -> bool {
        let all_set = Self::combination_all_set();
        let set_bit_count = (all_set & key).count_ones() as usize;

        L <= set_bit_count && set_bit_count <= U
    }

    const fn combination_all_set() -> usize {
        (2_u32.pow(N as u32) - 1) as usize
    }
}

impl<const L: usize, const U: usize, const N: usize, T> Drop for SubSetCache<L, U, N, T>
where
    [(); table_size(U, N)]:,
{
    fn drop(&mut self) {
        for i in L..=U {
            let all_set = Self::combination_all_set();
            for index in RawCombinationsIter::new(all_set, i as u32) {
                unsafe { self.table.get_unchecked_mut(index).assume_init_drop() }
            }
        }
    }
}

const fn factorial(n: usize) -> usize {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}

pub const fn combinations_exact(k: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    factorial(n) / (factorial(k) * factorial(n - k))
}

pub const fn combinations_from_to(mut lower: usize, upper: usize, n: usize) -> usize {
    let mut sum = 0;
    while upper >= lower {
        sum += combinations_exact(lower, n);
        lower += 1;
    }
    sum
}

pub const fn largest_index(upper: usize, n: usize) -> usize {
    let n_bits_set = 2_usize.pow(upper as u32) - 1;
    let shift = n - upper;

    n_bits_set << shift
}

pub const fn table_size(upper: usize, n: usize) -> usize {
    largest_index(upper, n) + 1
}

pub struct EntriesMutLUTIter<'s, const L: usize, const U: usize, const N: usize, T>
where
    [(); table_size(U, N)]:,
{
    it: std::slice::Iter<'s, usize>,
    ss_cache: &'s mut SubSetCache<L, U, N, T>,
}

impl<'s, const L: usize, const U: usize, const N: usize, T> Iterator for EntriesMutLUTIter<'s, L, U, N, T>
where
    [(); table_size(U, N)]:,
{
    type Item = (PencilMarks, &'s mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = *self.it.next()?;

        let elem = unsafe { &mut *self.ss_cache.table.get_unchecked_mut(index).as_mut_ptr() };

        // let elem = unsafe { &mut *self.ss_cache.table[index].as_mut_ptr() };
        let pm = PencilMarks::from_raw_bits(index as u16);
        Some((pm, elem))
    }
}

pub struct EntriesIter;

#[derive(Debug)]
pub struct EntriesMutIter<'s, const L: usize, const U: usize, const N: usize, T, E, Iter: Iterator<Item = E>>
where
    E: Deref<Target = usize>,
    [(); table_size(U, N)]:,
{
    k_cursor: u32,
    it: Iter,
    ss_cache: &'s mut SubSetCache<L, U, N, T>,
}

// impl<'s, const L: usize, const U: usize, const N: usize, T, E, Iter> Iterator
//     for EntriesMutIter<'s, L, U, N, T, E, Iter>
// where
//     E: Deref<Target = usize>,
//     Iter: Iterator<Item = E>,
//     [(); table_size(U, N)]:,
//     T: 's,
// {
//     type Item = (PencilMarks, &'s mut T);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.k_cursor as usize > U {
//             return None;
//         }

//         match self.it.next() {
//             Some(index) => {
//                 let elem = unsafe { &mut *self.ss_cache.table.get_unchecked_mut(index).as_mut_ptr() };
//                 let pm = PencilMarks::from_raw_bits(index as u16);
//                 Some((pm, elem))
//             }
//             None => {
//                 self.k_cursor += 1;

//                 let all_set = self.it.bits();

//                 self.it = RawCombinationsIter::new(all_set, self.k_cursor);
//                 self.next()
//             }
//         }
//     }
// }

// impl<'s, const L: usize, const U: usize, const N: usize, T> Iterator
//     for EntriesMutIter<'s, L, U, N, T, std::slice::Iter<'_, usize>>
// where
//     [(); table_size(U, N)]:,
//     T: 's,
// {
//     type Item = (PencilMarks, &'s mut T);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.k_cursor as usize > U {
//             return None;
//         }

//         match self.it.next() {
//             Some(index) => {
//                 let elem = unsafe { &mut *self.ss_cache.table.get_unchecked_mut(index).as_mut_ptr() };
//                 let pm = PencilMarks::from_raw_bits(index as u16);
//                 Some((pm, elem))
//             }
//             None => {
//                 self.k_cursor += 1;

//                 let all_set = self.it.bits();

//                 self.it = RawCombinationsIter::new(all_set, self.k_cursor);
//                 self.next()
//             }
//         }
//     }
// }

#[derive(Debug, Clone)]
pub struct EntriesExactIter<'s, const L: usize, const U: usize, const N: usize, T>
where
    [(); table_size(U, N)]:,
{
    it: RawCombinationsIter,
    ss_cache: &'s SubSetCache<L, U, N, T>,
}

impl<'a, const L: usize, const U: usize, const N: usize, T> Iterator for EntriesExactIter<'a, L, U, N, T>
where
    [(); table_size(U, N)]:,
{
    type Item = (PencilMarks, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.it.next()?;
        let elem = unsafe { &*self.ss_cache.table.get_unchecked(index).as_ptr() };
        Some((PencilMarks::from_raw_bits(index as u16), elem))
    }
}

#[derive(Debug)]
pub struct EntriesExactIterMut<'a, const L: usize, const U: usize, const N: usize, T>
where
    [(); table_size(U, N)]:,
{
    k: usize,
    cursor: usize,
    ss_cache: &'a mut SubSetCache<L, U, N, T>,
}

impl<'a, const L: usize, const U: usize, const N: usize, T> Iterator for EntriesExactIterMut<'a, L, U, N, T>
where
    [(); table_size(U, N)]:,
{
    type Item = (PencilMarks, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

pub struct PhfConstants {
    pub magic: usize,
    pub shift: usize,
}

pub const fn calculate_magic_and_shift(max_magic: usize, max_shift: usize) -> PhfConstants {
    let mut best_magic = 0;
    let mut best_shift = 0;

    let mut current_magic = 0;
    while current_magic <= max_magic {
        let mut current_shift = 0;
        while current_shift <= max_shift {
            current_shift += 1;
        }

        current_magic += 1;
    }
    let mut test = [false; 200];

    PhfConstants { magic: 0, shift: 0 }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn combinations_exact_1_16() {
        let n = combinations_exact(1, 9);
        assert_eq!(n, 9);
    }

    #[test]
    fn combinations_exact_2_16() {
        let n = combinations_exact(2, 9);
        assert_eq!(n, 36);
    }

    #[test]
    fn combinations_exact_4_16() {
        let n = combinations_exact(4, 9);
        assert_eq!(n, 126);
    }

    #[test]
    fn combinations_exact_0_0() {
        let n = combinations_exact(0, 0);
        assert_eq!(n, 0);
    }

    #[test]
    fn combinations_from_to_1_4_9() {
        let n = combinations_from_to(1, 4, 9);
        assert_eq!(n, 255);
    }

    #[test]
    fn combinations_from_to_2_4_9() {
        let n = combinations_from_to(2, 4, 9);
        assert_eq!(n, 246);
    }

    #[test]
    fn combinations_from_to_3_4_9() {
        let n = combinations_from_to(3, 4, 9);
        assert_eq!(n, 210);
    }

    #[test]
    fn largest_index_2_9() {
        let max_index = largest_index(2, 9);
        assert_eq!(max_index, 0b001_1000_0000);
    }

    #[test]
    fn largest_index_3_9() {
        let max_index = largest_index(3, 9);
        assert_eq!(max_index, 0b001_1100_0000);
    }

    #[test]
    fn largest_index_0_9() {
        let max_index = largest_index(0, 9);
        assert_eq!(max_index, 0);
    }

    #[test]
    fn table_size_2_9() {
        let table_size = table_size(2, 9);
        assert_eq!(table_size, 385);
    }

    #[test]
    fn table_size_0_9() {
        let table_size = table_size(0, 9);
        assert_eq!(table_size, 1);
    }
}
