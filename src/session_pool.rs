//! Kept server sessions. Each request runs in the kept session whose tokens
//! are the longest strict prefix of its prompt, so interleaved conversations
//! do not evict each other; otherwise it starts fresh and, when the pool is
//! full, the least recently used session is given up for it.
use crate::engine::{Engine, Session};
use crate::{Result, invalid};

/// Kept sessions when `--sessions` is omitted and memory allows: a coding
/// agent's main conversation plus its interleaved side requests (titles,
/// summaries, a subagent) without re-prefilling the main prompt.
pub const DEFAULT_SESSIONS: usize = 4;

/// Sessions that fit the budget. The engine reservation already holds one
/// state; each further session needs `state` bytes of the remaining budget.
/// An explicit request that does not fit is an error; the default shrinks.
pub fn capacity(
    requested: Option<usize>,
    budget: usize,
    reserved: usize,
    state: usize,
) -> Result<usize> {
    let free = budget.saturating_sub(reserved);
    let fits = 1 + free.checked_div(state).unwrap_or(usize::MAX - 1);
    match requested {
        None => Ok(DEFAULT_SESSIONS.min(fits)),
        Some(0) => Err(invalid("--sessions must be at least 1")),
        Some(n) if n <= fits => Ok(n),
        Some(n) => {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            Err(invalid(format!(
                "--sessions {n} needs {:.1} GiB for {} more session states of {:.1} GiB, \
                 but the memory budget has {:.1} GiB left after the model ({fits} session(s) fit); \
                 lower --sessions or CONTEXT, or raise MEMORY_GIB",
                gib((n - 1) * state),
                n - 1,
                gib(state),
                gib(free),
            )))
        }
    }
}

/// Size the pool from one measured session state and let the engine keep
/// that many live sessions. Returns the session count and state bytes.
pub fn configure(
    engine: &mut Engine,
    requested: Option<usize>,
    budget: usize,
) -> Result<(usize, usize)> {
    let state = engine.session_bytes()?;
    let n = capacity(requested, budget, engine.reserved_bytes(), state)?;
    engine.set_max_sessions(n);
    Ok((n, state))
}

pub trait Tokens {
    fn tokens(&self) -> &[u32];
}
impl Tokens for Session {
    fn tokens(&self) -> &[u32] {
        self.consumed()
    }
}

struct Slot<S> {
    id: u64,
    used: u64,
    session: S,
}

/// What `take` chose for a prompt.
#[derive(Debug, PartialEq, Eq)]
pub enum Pick {
    /// Kept session `id` holds `tokens` of the prompt.
    Reuse { id: u64, tokens: usize },
    /// No kept session is a prefix; `evicted` was the least recently used.
    Fresh { evicted: Option<u64> },
}

impl Pick {
    pub fn reused(&self) -> Option<u64> {
        match self {
            Self::Reuse { id, .. } => Some(*id),
            Self::Fresh { .. } => None,
        }
    }
}

/// Log text: which session ran the request and what happened to the pool.
pub fn describe(pick: &Pick, kept: Option<u64>, len: usize) -> String {
    let kept = kept.map_or("not kept".to_owned(), |id| format!("kept as #{id}"));
    match pick {
        Pick::Reuse { id, tokens } => format!("#{id}(reused {tokens}, {kept}, {len} kept)"),
        Pick::Fresh { evicted: Some(e) } => format!("new(evicted #{e}, {kept}, {len} kept)"),
        Pick::Fresh { evicted: None } => format!("new({kept}, {len} kept)"),
    }
}

pub struct Pool<S> {
    slots: Vec<Slot<S>>,
    capacity: usize,
    clock: u64,
    next_id: u64,
    partial: bool,
}
impl<S: Tokens> Pool<S> {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![],
            capacity: capacity.max(1),
            clock: 0,
            next_id: 0,
            partial: false,
        }
    }
    /// Allow reusing a session that shares only part of a prompt (models whose
    /// state can be truncated: attention-only KV caches).
    pub fn partial(mut self, yes: bool) -> Self {
        self.partial = yes;
        self
    }
    /// Prompt tokens a kept session can serve: its whole length when it is a
    /// strict prefix, or with partial reuse the common prefix when it covers at
    /// least half the session (re-rendered chat turns differ only in the tail;
    /// a short accidental overlap would discard a useful cache).
    fn usable(&self, t: &[u32], prompt: &[u32]) -> usize {
        if t.is_empty() || prompt.len() < 2 {
            return 0;
        }
        if t.len() < prompt.len() && prompt.starts_with(t) {
            return t.len();
        }
        if !self.partial {
            return 0;
        }
        let common = t.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let common = common.min(prompt.len() - 1);
        if common > 0 && common * 2 >= t.len() {
            common
        } else {
            0
        }
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Remove the session to run `prompt` in. A reused session is returned for
    /// the engine to extend. Otherwise, when the pool is full, the least
    /// recently used one is returned too: the engine drops a non-prefix session
    /// before allocating, so memory stays within `capacity` states.
    pub fn take(&mut self, prompt: &[u32]) -> (Option<S>, Pick) {
        let best = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| (i, self.usable(s.session.tokens(), prompt), s.used))
            .filter(|&(_, usable, _)| usable > 0)
            .max_by_key(|&(_, usable, used)| (usable, used))
            .map(|(i, usable, _)| (i, usable));
        if let Some((i, tokens)) = best {
            let slot = self.slots.swap_remove(i);
            return (
                Some(slot.session),
                Pick::Reuse {
                    id: slot.id,
                    tokens,
                },
            );
        }
        if self.slots.len() < self.capacity {
            return (None, Pick::Fresh { evicted: None });
        }
        let i = (0..self.slots.len())
            .min_by_key(|&i| self.slots[i].used)
            .unwrap();
        let slot = self.slots.swap_remove(i);
        (
            Some(slot.session),
            Pick::Fresh {
                evicted: Some(slot.id),
            },
        )
    }
    /// Keep a session after its request; `id` is the reused slot, or None for
    /// a new one. Returns the id it is kept under.
    pub fn put(&mut self, id: Option<u64>, session: S) -> u64 {
        let id = id.unwrap_or_else(|| {
            self.next_id += 1;
            self.next_id
        });
        self.clock += 1;
        while self.slots.len() >= self.capacity {
            let i = (0..self.slots.len())
                .min_by_key(|&i| self.slots[i].used)
                .unwrap();
            self.slots.swap_remove(i);
        }
        self.slots.push(Slot {
            id,
            used: self.clock,
            session,
        });
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    impl Tokens for Vec<u32> {
        fn tokens(&self) -> &[u32] {
            self
        }
    }

    #[test]
    fn partial_reuse_needs_half_the_session_shared() {
        let base: Vec<u32> = (0..100).collect();
        let mut rerendered = base[..80].to_vec();
        rerendered.extend([500, 501, 502]);
        let mut strict = Pool::new(2);
        strict.put(None, base.clone());
        assert_eq!(strict.take(&rerendered).1, Pick::Fresh { evicted: None });
        let mut pool = Pool::new(2).partial(true);
        let id = pool.put(None, base.clone());
        let (s, pick) = pool.take(&rerendered);
        assert_eq!(s.unwrap(), base);
        assert_eq!(pick, Pick::Reuse { id, tokens: 80 });
        // A short overlap leaves the cache alone.
        pool.put(Some(id), base.clone());
        let mut short = base[..10].to_vec();
        short.push(999);
        assert_eq!(pool.take(&short).1, Pick::Fresh { evicted: None });
        // An identical prompt reuses all but the last token.
        let (_, pick) = pool.take(&base);
        assert_eq!(pick, Pick::Reuse { id, tokens: 99 });
    }

    #[test]
    fn interleaved_request_keeps_the_long_conversation() {
        let mut pool = Pool::new(2);
        let a: Vec<u32> = (0..100).collect();
        let (s, pick) = pool.take(&a);
        assert!(s.is_none());
        assert_eq!(pick, Pick::Fresh { evicted: None });
        let ida = pool.put(None, a.clone());
        let b = vec![900, 901, 902];
        assert_eq!(pool.take(&b).1, Pick::Fresh { evicted: None });
        let idb = pool.put(None, b);
        assert_ne!(ida, idb);
        let mut extended = a.clone();
        extended.extend([100, 101]);
        let (s, pick) = pool.take(&extended);
        assert_eq!(s.unwrap(), a);
        assert_eq!(
            pick,
            Pick::Reuse {
                id: ida,
                tokens: 100
            }
        );
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn longest_strict_prefix_wins() {
        let mut pool = Pool::new(4);
        let short = pool.put(None, vec![1, 2]);
        let long = pool.put(None, vec![1, 2, 3, 4]);
        pool.put(None, vec![1, 2, 3, 4, 5, 6]); // equals the prompt: not strict
        pool.put(None, vec![1, 9]);
        let (s, pick) = pool.take(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(s.unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(
            pick,
            Pick::Reuse {
                id: long,
                tokens: 4
            }
        );
        let (_, pick) = pool.take(&[1, 2, 7]);
        assert_eq!(
            pick,
            Pick::Reuse {
                id: short,
                tokens: 2
            }
        );
    }

    #[test]
    fn full_pool_hands_out_the_least_recently_used() {
        let mut pool = Pool::new(2);
        let first = pool.put(None, vec![1]);
        let second = pool.put(None, vec![2]);
        // Reusing `first` makes `second` the least recently used.
        let (s, _) = pool.take(&[1, 1]);
        pool.put(Some(first), s.unwrap());
        let (s, pick) = pool.take(&[3, 3]);
        assert_eq!(s.unwrap(), vec![2]);
        assert_eq!(
            pick,
            Pick::Fresh {
                evicted: Some(second)
            }
        );
        let third = pool.put(None, vec![3, 3, 4]);
        assert_eq!(pool.len(), 2);
        assert!(third != first && third != second);
    }

    #[test]
    fn put_never_exceeds_capacity() {
        let mut pool = Pool::new(1);
        pool.put(None, vec![1]);
        pool.put(None, vec![2]);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.take(&[2, 0]).0.unwrap(), vec![2]);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn capacity_comes_from_the_remaining_budget() {
        let gib = 1usize << 30;
        assert_eq!(capacity(None, 80 * gib, 30 * gib, 10 * gib).unwrap(), 4);
        assert_eq!(capacity(None, 80 * gib, 55 * gib, 10 * gib).unwrap(), 3);
        assert_eq!(capacity(None, 80 * gib, 79 * gib, 10 * gib).unwrap(), 1);
        assert_eq!(capacity(Some(6), 80 * gib, 30 * gib, 10 * gib).unwrap(), 6);
        let e = capacity(Some(7), 80 * gib, 30 * gib, 10 * gib).unwrap_err();
        assert!(e.to_string().contains("6 session(s) fit"), "{e}");
        assert!(capacity(Some(0), 80 * gib, 30 * gib, 10 * gib).is_err());
    }
}
