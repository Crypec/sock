use crate::board::PencilMarks;

// L: Lower bound for subsets length
// U: Upper bound for subsets length
// N: Size of set
#[derive(Debug, Clone)]
pub struct SubSetCache<const L: usize, const U: usize, const N: usize, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    table: [T; combinations_from_to(L, U, N)],
}

impl<const N: usize, const L: usize, const U: usize, T> SubSetCache<L, U, N, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    pub fn from_array(table: [T; combinations_from_to(L, U, N)]) -> Self {
        Self { table }
    }

    pub fn get(&self, key: usize) -> Option<&T> {
        self.table.get(key)
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        self.table.get_mut(key)
    }

    pub unsafe fn get_unchecked(&self, key: usize) -> &T {
        debug_assert!(key < self.table.len());
        self.table.get_unchecked(key)
    }

    pub unsafe fn get_unchecked_mut(&mut self, key: usize) -> &mut T {
        debug_assert!(key < self.table.len());
        self.table.get_unchecked_mut(key)
    }

    pub fn entries_exact(&self, k: usize) -> EntriesExactIter<L, U, N, T> {
        EntriesExactIter {
            k,
            cursor: 0,
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

pub const fn combinations_from_to(lower: usize, upper: usize, n: usize) -> usize {
    const fn inner(i: usize, upper: usize, n: usize) -> usize {
        if i == upper {
            return combinations_exact(i, n);
        }
        combinations_exact(i, n) + inner(i + 1, upper, n)
    }

    inner(lower, upper, n)
}

#[derive(Debug, Clone)]
pub struct EntriesExactIter<'a, const L: usize, const U: usize, const N: usize, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    k: usize,
    cursor: usize,
    ss_cache: &'a SubSetCache<L, U, N, T>,
}

impl<'a, const L: usize, const U: usize, const N: usize, T> Iterator for EntriesExactIter<'a, L, U, N, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    type Item = (PencilMarks, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        todo!();
    }
}

#[derive(Debug)]
pub struct EntriesExactIterMut<'a, const L: usize, const U: usize, const N: usize, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    k: usize,
    cursor: usize,
    ss_cache: &'a mut SubSetCache<L, U, N, T>,
}

impl<'a, const L: usize, const U: usize, const N: usize, T> Iterator for EntriesExactIterMut<'a, L, U, N, T>
where
    [(); combinations_from_to(L, U, N)]:,
{
    type Item = (PencilMarks, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
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
}
