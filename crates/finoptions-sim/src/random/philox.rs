//! Minimal Random123-compatible Philox4x32-10 core with disjoint path namespaces.

const MULTIPLIER_0: u32 = 0xd251_1f53;
const MULTIPLIER_1: u32 = 0xcd9e_8d57;
const KEY_INCREMENT_0: u32 = 0x9e37_79b9;
const KEY_INCREMENT_1: u32 = 0xbb67_ae85;

#[inline]
fn multiply_high_low(left: u32, right: u32) -> (u32, u32) {
    let product = u64::from(left) * u64::from(right);
    (
        u32::try_from(product >> 32).unwrap(),
        u32::try_from(product & u64::from(u32::MAX)).unwrap(),
    )
}

#[inline]
fn round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (high_0, low_0) = multiply_high_low(MULTIPLIER_0, counter[0]);
    let (high_1, low_1) = multiply_high_low(MULTIPLIER_1, counter[2]);
    [
        high_1 ^ counter[1] ^ key[0],
        low_1,
        high_0 ^ counter[3] ^ key[1],
        low_0,
    ]
}

/// Random123 Philox4x32 with exactly ten rounds.
#[inline]
#[must_use]
pub(crate) fn philox4x32_10(counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    let mut output = round(counter, key);
    for _ in 1..10 {
        key[0] = key[0].wrapping_add(KEY_INCREMENT_0);
        key[1] = key[1].wrapping_add(KEY_INCREMENT_1);
        output = round(output, key);
    }
    output
}

#[inline]
fn split_u64(word: u64) -> [u32; 2] {
    [
        u32::try_from(word & u64::from(u32::MAX)).unwrap(),
        u32::try_from(word >> 32).unwrap(),
    ]
}

/// One path-local stream identified injectively by `(seed, domain, path)`.
#[derive(Clone, Debug)]
pub(crate) struct PhiloxPathStream {
    key: [u32; 2],
    domain: u32,
    path: u32,
    next_block: u64,
    remaining_normals: u32,
    buffer: [u32; 4],
    buffer_position: usize,
}

impl PhiloxPathStream {
    #[inline]
    pub(crate) fn new(seed: u64, domain: u32, path: u32, normal_budget: u32) -> Self {
        Self {
            key: split_u64(seed),
            domain,
            path,
            next_block: 0,
            remaining_normals: normal_budget,
            buffer: [0; 4],
            buffer_position: 4,
        }
    }

    #[inline]
    fn refill(&mut self) {
        // The nested bounded rejection paths admit at most 2^21 blocks per normal and construction
        // admits at most u32::MAX normals. Their product is below 2^53, so this 64-bit block lane
        // cannot wrap.
        debug_assert_ne!(self.next_block, u64::MAX);
        let block = split_u64(self.next_block);
        self.buffer = philox4x32_10([block[0], block[1], self.path, self.domain], self.key);
        self.buffer_position = 0;
        self.next_block += 1;
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.buffer_position == self.buffer.len() {
            self.refill();
        }
        let word = self.buffer[self.buffer_position];
        self.buffer_position += 1;
        word
    }

    #[inline]
    pub(crate) const fn begin_normal(&mut self) -> bool {
        if self.remaining_normals == 0 {
            false
        } else {
            self.remaining_normals -= 1;
            true
        }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32());
        low | (high << 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_random123_known_answer_vectors() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn domain_and_path_occupy_distinct_counter_lanes() {
        let mut left = PhiloxPathStream::new(7, 11, 13, 2);
        let mut other_path = PhiloxPathStream::new(7, 11, 14, 1);
        let mut other_domain = PhiloxPathStream::new(7, 12, 13, 1);
        assert_ne!(left.next_u64(), other_path.next_u64());
        assert_ne!(left.next_u64(), other_domain.next_u64());
    }

    #[test]
    fn enforces_the_declared_normal_budget_before_sampling() {
        let mut stream = PhiloxPathStream::new(7, 11, 13, 1);
        assert!(stream.begin_normal());
        assert!(!stream.begin_normal());
    }
}
