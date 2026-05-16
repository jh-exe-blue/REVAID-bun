use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit, size_of};
use core::ptr::NonNull;

use bun_core::asan;

/// Fixed-width occupancy bitset for [`HiveArray`].
///
/// PORT NOTE: Zig's `std.bit_set.IntegerBitSet(N)` is backed by an exact-width
/// `uN` integer (`u128`, `u256`, `u2048`, …). The Rust port's
/// [`IntegerBitSet`](crate::bit_set::IntegerBitSet) is backed by a single
/// `usize`, so for `N > 64` it silently held only 64 usable bits — every
/// `HiveArray<_, 128/256/2048>` pool degraded to 64 effective slots and spilled
/// to the heap fallback on the 65th in-flight item. Under HTTP load (the
/// `Body::Value` 256-slot pool, the `RequestContext` 2048-slot pool) this turned
/// every request into a `Box::new`.
///
/// We can't spell `[usize; (CAPACITY+63)/64]` without `generic_const_exprs`
/// (which would virally add `where` bounds on every `HiveArray` consumer), so
/// this uses a fixed `[usize; 32]` backing array — 2048 bits, which is the
/// largest in-tree `HiveArray` capacity. Only the first
/// `ceil(CAPACITY/64)` words are touched, so smaller pools pay 256 B of dead
/// storage (negligible next to `buffer: [MaybeUninit<T>; CAPACITY]`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HiveBitSet<const CAPACITY: usize> {
    masks: [usize; HIVE_BITSET_WORDS],
}

const HIVE_BITSET_WORDS: usize = 32;
const WORD_BITS: usize = usize::BITS as usize;

impl<const CAPACITY: usize> HiveBitSet<CAPACITY> {
    const NUM_WORDS: usize = if CAPACITY == 0 {
        0
    } else {
        (CAPACITY + WORD_BITS - 1) / WORD_BITS
    };
    const _FITS: () = assert!(
        CAPACITY <= HIVE_BITSET_WORDS * WORD_BITS,
        "HiveArray CAPACITY exceeds HiveBitSet backing (raise HIVE_BITSET_WORDS)"
    );
    /// Mask of valid bits in the last live word (all-ones when CAPACITY is a
    /// multiple of 64; otherwise zeros in the high padding bits).
    const LAST_WORD_MASK: usize = {
        let rem = CAPACITY % WORD_BITS;
        if rem == 0 {
            usize::MAX
        } else {
            (1usize << rem) - 1
        }
    };

    pub const fn init_empty() -> Self {
        Self {
            masks: [0; HIVE_BITSET_WORDS],
        }
    }

    #[inline]
    pub fn is_set(&self, index: usize) -> bool {
        debug_assert!(index < CAPACITY);
        (self.masks[index / WORD_BITS] >> (index % WORD_BITS)) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        debug_assert!(index < CAPACITY);
        self.masks[index / WORD_BITS] |= 1usize << (index % WORD_BITS);
    }

    #[inline]
    pub fn unset(&mut self, index: usize) {
        debug_assert!(index < CAPACITY);
        self.masks[index / WORD_BITS] &= !(1usize << (index % WORD_BITS));
    }

    #[inline]
    pub fn find_first_set(&self) -> Option<usize> {
        let mut i = 0;
        while i < Self::NUM_WORDS {
            let m = self.masks[i];
            if m != 0 {
                return Some(i * WORD_BITS + m.trailing_zeros() as usize);
            }
            i += 1;
        }
        None
    }

    #[inline]
    pub fn find_first_unset(&self) -> Option<usize> {
        let mut i = 0;
        while i < Self::NUM_WORDS {
            let live_mask = if i + 1 == Self::NUM_WORDS {
                Self::LAST_WORD_MASK
            } else {
                usize::MAX
            };
            let inv = !self.masks[i] & live_mask;
            if inv != 0 {
                return Some(i * WORD_BITS + inv.trailing_zeros() as usize);
            }
            i += 1;
        }
        None
    }

    /// Forward iterator over set bits. Mirrors `IntegerBitSet::iter_set`.
    #[inline]
    pub fn iter_set(&self) -> HiveBitSetIter<CAPACITY> {
        self.iterator::<true, true>()
    }

    /// Signature mirrors `IntegerBitSet::iterator` so existing
    /// `hive.used.iterator::<true, true>()` callers compile unchanged. Only
    /// the `<KIND_SET=true, DIR_FWD=true>` combination is implemented (the
    /// only one used in-tree); other params assert.
    #[inline]
    pub fn iterator<const KIND_SET: bool, const DIR_FWD: bool>(&self) -> HiveBitSetIter<CAPACITY> {
        const {
            assert!(
                KIND_SET && DIR_FWD,
                "HiveBitSet::iterator only supports <true,true>"
            )
        };
        HiveBitSetIter {
            masks: self.masks,
            word: 0,
        }
    }
}

pub struct HiveBitSetIter<const CAPACITY: usize> {
    masks: [usize; HIVE_BITSET_WORDS],
    word: usize,
}

impl<const CAPACITY: usize> HiveBitSetIter<CAPACITY> {
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<usize> {
        while self.word < HiveBitSet::<CAPACITY>::NUM_WORDS {
            let m = self.masks[self.word];
            if m != 0 {
                let bit = m.trailing_zeros() as usize;
                self.masks[self.word] &= m - 1;
                return Some(self.word * WORD_BITS + bit);
            }
            self.word += 1;
        }
        None
    }
}

/// An array that efficiently tracks which elements are in use.
/// The pointers are intended to be stable
/// Sorta related to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0447r15.html
// PORT NOTE: Zig's `capacity: u16` is widened to `usize` here because Rust array
// lengths require a `usize` const generic on stable.
pub struct HiveArray<T, const CAPACITY: usize> {
    pub buffer: [MaybeUninit<T>; CAPACITY],
    pub used: HiveBitSet<CAPACITY>,
}

impl<T, const CAPACITY: usize> HiveArray<T, CAPACITY> {
    pub const SIZE: usize = CAPACITY;

    // PORT NOTE: Zig had `pub var empty: Self` as a mutable static to work around
    // https://github.com/ziglang/zig/issues/22462 and /21988. Rust has no such
    // limitation; callers should use `init()` (which is `const`).

    pub const fn init() -> Self {
        Self {
            buffer: [const { MaybeUninit::uninit() }; CAPACITY],
            used: HiveBitSet::init_empty(),
        }
    }

    /// Placement-new constructor: write the empty state directly into `*out`
    /// without materializing `Self` on the stack.
    ///
    /// `Self` embeds `[MaybeUninit<T>; CAPACITY]` inline, which for the
    /// install pools (`NetworkTask` × 128, `Task` × 64) is hundreds of KB.
    /// Rust has no result-location semantics, so `out.write(Self::init())`
    /// first builds the value in the caller's frame and `memcpy`s it — LLVM
    /// does **not** elide that temporary. This entry point only writes the
    /// 256 B `used` bitset; `buffer` is `MaybeUninit` and needs no
    /// initialization (uninitialized bytes are a valid bit-pattern for it).
    ///
    /// # Safety
    /// `out` must be non-null, properly aligned, and valid for writes of
    /// `size_of::<Self>()` bytes. The previous contents are not dropped.
    #[inline]
    pub unsafe fn init_in_place(out: *mut Self) {
        // SAFETY: caller contract — `out` is aligned and writable. We form a
        // place expression on `*out` only to project to `used`; no `&mut Self`
        // is created over the (uninitialized) whole struct.
        unsafe {
            core::ptr::addr_of_mut!((*out).used).write(HiveBitSet::init_empty());
        }
        // `buffer: [MaybeUninit<T>; CAPACITY]` intentionally untouched.
    }

    /// One-shot claim + write. Preferred entry point — no uninit window.
    ///
    /// Returns `None` (and does **not** consume `value`'s slot) if the hive
    /// is full; on `None` the caller still owns `value` and must drop it.
    #[inline]
    pub fn get_init(&mut self, value: T) -> Option<NonNull<T>> {
        Some(self.claim()?.write(value))
    }

    /// Claim + write where `T` is self-referential on its own slot address
    /// (e.g. a struct that registers itself as a uws/libuv user-data pointer
    /// inside its own constructor). `init` receives the slot's stable address
    /// and must return the value to be stored there.
    #[inline]
    pub fn emplace(&mut self, init: impl FnOnce(NonNull<T>) -> T) -> Option<NonNull<T>> {
        let slot = self.claim()?;
        let addr = slot.addr();
        Some(slot.write(init(addr)))
    }

    /// Low-level reservation. Only when [`get_init`](Self::get_init) /
    /// [`emplace`](Self::emplace) are insufficient — typically when the caller
    /// must interleave fallible work between claim and commit, or perform
    /// `repr(C)` placement-new via [`HiveSlot::as_uninit`].
    ///
    /// The returned token borrows `self` for `'_`; precompute any raw
    /// back-pointers to the parent struct *before* calling `claim()` if they
    /// are needed inside the initializer.
    pub fn claim(&mut self) -> Option<HiveSlot<'_, T, CAPACITY>> {
        let index = self.used.find_first_unset()?;
        self.used.set(index);
        let slot = NonNull::from(&mut self.buffer[index]);
        asan::unpoison(slot.as_ptr().cast(), size_of::<T>());
        let owner = core::ptr::from_mut(self) as usize;
        // Tagged-pointer scheme requires the low bit clear for inline slots.
        // `HiveArray` is at least pointer-aligned via `IntegerBitSet`'s
        // backing word, and in practice `align_of::<T>() >= 2` for every `T`
        // we pool; assert in debug so a future 1-byte `T` is caught.
        debug_assert_eq!(
            owner & 1,
            0,
            "HiveArray must be >=2-byte aligned for HiveSlot owner tag"
        );
        Some(HiveSlot {
            slot,
            owner,
            _marker: PhantomData,
        })
    }

    /// Recycle a slot **without** running `T::drop`. Safe: if `value` does not
    /// point into this hive, returns `false` and is a no-op. Use when the
    /// caller has already moved the contents out / destructured them, or when
    /// `T` is POD and the slot is being released on an error path before it
    /// was fully initialized (Zig `value.* = undefined`).
    pub fn put_raw(&mut self, value: *mut T) -> bool {
        let Some(index) = self.index_of(value) else {
            return false;
        };
        debug_assert!(self.used.is_set(index as usize));
        asan::poison(value.cast(), size_of::<T>());
        self.used.unset(index as usize);
        true
    }

    pub fn at(&mut self, index: u16) -> *mut T {
        debug_assert!((index as usize) < CAPACITY);
        let ret = self.buffer[index as usize].as_mut_ptr();
        asan::assert_unpoisoned(ret.cast::<u8>());
        ret
    }

    pub fn index_of(&self, value: *const T) -> Option<u32> {
        asan::assert_unpoisoned(value.cast::<u8>());
        let start = self.buffer.as_ptr().cast::<T>();
        // One-past-the-end pointer of `buffer`; `wrapping_add` is sound for
        // the in-allocation offset and matches `add` exactly here.
        let end = start.wrapping_add(CAPACITY);
        if !((value as usize) >= (start as usize) && (value as usize) < (end as usize)) {
            return None;
        }

        // aligned to the size of T
        let index = ((value as usize) - (start as usize)) / size_of::<T>();
        debug_assert!(index < CAPACITY);
        debug_assert!(self.buffer[index].as_ptr().cast::<T>() == value);
        Some(u32::try_from(index).expect("int cast"))
    }

    pub fn r#in(&self, value: *const T) -> bool {
        asan::assert_unpoisoned(value.cast::<u8>());
        let start = self.buffer.as_ptr().cast::<T>();
        let end = start.wrapping_add(CAPACITY);
        (value as usize) >= (start as usize) && (value as usize) < (end as usize)
    }

    /// Return a slot to the pool, dropping the contained `T` in place.
    ///
    /// Returns `false` (and drops nothing) if `value` does not point into
    /// this hive's buffer.
    ///
    /// # Safety
    /// If `value` points into this hive, it must point to a fully-initialized
    /// `T` previously obtained via [`get_init`](Self::get_init) /
    /// [`emplace`](Self::emplace), or via [`claim`](Self::claim) followed by
    /// [`HiveSlot::write`] / [`HiveSlot::assume_init`]. The slot is dropped in
    /// place; passing a moved-from or uninitialized slot is UB for `T` with
    /// drop glue.
    pub unsafe fn put(&mut self, value: *mut T) -> bool {
        let Some(index) = self.index_of(value) else {
            return false;
        };

        debug_assert!(self.used.is_set(index as usize));
        debug_assert!(self.buffer[index as usize].as_ptr().cast::<T>() == value.cast_const());

        // PORT NOTE: Zig wrote `value.* = undefined;` — Zig has no destructors,
        // so the slot was simply marked logically uninitialized. In the Rust
        // port several `T` carry owned heap data (e.g. `NumberScope.name_counts:
        // StringHashMap`, `NetworkTask.url_buf: Box<[u8]>`); drop the slot
        // before recycling so the put/get cycle does not leak it. Callers that
        // pre-clean fields (`PooledSocket::release_parked_refs`) leave only
        // trivially-droppable residuals, so this is idempotent for them.
        // SAFETY: caller contract — `value` is a fully-initialized `T` in `buffer`.
        unsafe { core::ptr::drop_in_place(value) };
        asan::poison(value.cast(), size_of::<T>());

        self.used.unset(index as usize);
        true
    }
}

// ──────────────────────────────────────────────────────────────────────────
// HiveSlot
// ──────────────────────────────────────────────────────────────────────────

/// Linear reservation token for a claimed-but-uninitialized hive slot.
///
/// `HiveArray` slots are `[MaybeUninit<T>; CAP]`. The legacy `get()`
/// contract was two-phase — claim a `*mut T` to garbage, then `ptr::write` it
/// — which opened three UB hazards in the gap: (H1) early-return / `?` / panic
/// leaves the slot claimed-uninit so a later `put()` drops garbage; (H2)
/// `&mut *p` over uninit `T` is instant validity UB when `T` has niches; (H3)
/// partial field-write then `assume_init_ref` on the whole slot.
///
/// `HiveSlot` encodes the invariant **"a `used` slot is always fully
/// initialized"** in the type system: you cannot obtain the stable
/// initialized `*mut T` without going through [`write`](Self::write) (or the
/// `unsafe` [`assume_init`](Self::assume_init) escape hatch). If the token is
/// dropped (early return, `?`, panic) the slot is released **without** running
/// `T::drop` — it was never written.
///
/// Two-pointer-sized; `owner` is a tagged `usize`:
///   - low bit `0` ⇒ `*mut HiveArray<T, CAP>` (release = unset `used` bit + poison),
///   - low bit `1` ⇒ heap `Box<MaybeUninit<T>>` (release = dealloc, no `T::drop`).
///
/// **Aliasing note** (matches the `BackRef<T>` precedent in `bun_ptr`): the
/// token stores a raw `*mut HiveArray` rather than `&'h mut HiveArray`. The
/// `PhantomData<&'h mut _>` keeps it lifetime-scoped to the `claim()` borrow,
/// but the structural guarantee — the hive is a field of a long-lived owner
/// that is not moved between `claim()` and `write()` — is the caller's, same
/// as every back-pointer in the port.
#[must_use = "claimed hive slot is leaked if neither written nor dropped"]
pub struct HiveSlot<'h, T, const CAPACITY: usize> {
    slot: NonNull<MaybeUninit<T>>,
    /// Tagged owner; see type-level docs.
    owner: usize,
    _marker: PhantomData<&'h mut HiveArray<T, CAPACITY>>,
}

impl<'h, T, const CAPACITY: usize> HiveSlot<'h, T, CAPACITY> {
    /// Stable address of the slot. Safe to capture (e.g. register as a
    /// libuv/uws user-data pointer) **before** [`write`](Self::write), as long
    /// as nothing dereferences it until after `write()`.
    #[inline]
    pub fn addr(&self) -> NonNull<T> {
        self.slot.cast::<T>()
    }

    /// `&mut MaybeUninit<T>` for piecewise init via `addr_of_mut!`. Prefer
    /// [`write`](Self::write); this exists for `repr(C)` placement-new
    /// (`create_in`-style constructors that take `&mut MaybeUninit<Self>`).
    #[inline]
    pub fn as_uninit(&mut self) -> &mut MaybeUninit<T> {
        // SAFETY: `slot` is a unique live pointer into the hive buffer (or a
        // freshly leaked `Box<MaybeUninit<T>>`); the `&mut self` receiver
        // guarantees no other `&mut` to the same `MaybeUninit<T>` exists.
        unsafe { self.slot.as_mut() }
    }

    /// Move `value` into the slot and return the stable initialized pointer.
    /// Consumes the token (its `Drop` does not run).
    #[inline]
    pub fn write(self, value: T) -> NonNull<T> {
        let mut this = ManuallyDrop::new(self);
        NonNull::from(this.as_uninit().write(value))
    }

    /// Caller has fully initialized the slot via [`as_uninit`](Self::as_uninit)
    /// (or by writing through [`addr`](Self::addr)). Consumes the token.
    ///
    /// # Safety
    /// Every field of `T` must be initialized, including padding-adjacent
    /// niches (enum discriminants, `NonNull`, `Box`, `&`). Calling this on a
    /// partially-written slot is the exact UB this type exists to prevent.
    #[inline]
    pub unsafe fn assume_init(self) -> NonNull<T> {
        let this = ManuallyDrop::new(self);
        this.slot.cast::<T>()
    }

    /// Consume the token into an owned, lifetime-erased [`HiveOwned`] handle.
    /// The slot stays claimed (the token's `Drop` does not run); recycle later
    /// via [`Fallback::put`] / [`Fallback::put_raw`].
    ///
    /// `HiveOwned` does not deref by itself, so this does not require the slot
    /// to already be initialized — that obligation moves to the first
    /// [`HiveOwned::as_ref`] / [`HiveOwned::as_mut`].
    ///
    /// # Safety
    /// The slot must be fully initialized before any call to
    /// [`HiveOwned::as_ref`] / [`HiveOwned::as_mut`] on the returned handle.
    /// Use after [`write`](Self::write) or after placement-init through
    /// [`addr`](Self::addr) / [`as_uninit`](Self::as_uninit).
    #[inline]
    pub unsafe fn into_owned(self) -> HiveOwned<T> {
        let this = ManuallyDrop::new(self);
        HiveOwned(this.slot.cast::<T>())
    }
}

impl<T, const CAPACITY: usize> Drop for HiveSlot<'_, T, CAPACITY> {
    fn drop(&mut self) {
        if self.owner & 1 == 0 {
            // Inline hive slot: unset the `used` bit and re-poison. Do NOT
            // `drop_in_place` — the slot was never `.write()`n.
            let hive = self.owner as *mut HiveArray<T, CAPACITY>;
            // SAFETY: `owner` was set from `core::ptr::from_mut(self)` in
            // `HiveArray::claim`; the hive is a field of a long-lived owner
            // that has not been moved (structural back-pointer guarantee).
            // No `&mut HiveArray` is live across this drop — `claim()`'s
            // borrow was released when the raw pointer was captured.
            unsafe {
                let index = (*hive)
                    .index_of(self.slot.as_ptr().cast::<T>())
                    .expect("HiveSlot points outside its owning hive");
                asan::poison(self.slot.as_ptr().cast(), size_of::<T>());
                (*hive).used.unset(index as usize);
            }
        } else {
            // Heap fallback slot: reclaim the `Box<MaybeUninit<T>>` allocation.
            // `MaybeUninit<T>` has no drop glue, so this deallocates without
            // touching `T`.
            // SAFETY: `slot` was produced by `Box::leak(Box::<MaybeUninit<T>>::new_uninit())`
            // in `Fallback::claim` and has not been freed.
            drop(unsafe { Box::from_raw(self.slot.as_ptr()) });
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// HiveOwned
// ──────────────────────────────────────────────────────────────────────────

/// Owned, lifetime-erased handle to a fully-initialized pool slot.
///
/// Constructible only from a real pool claim (no public `new`), so safe code
/// cannot manufacture one from a dangling pointer. Move-only — there is exactly
/// one per claimed slot, so aliasing two of them is a compile error.
///
/// Unlike [`HiveSlot`], this does not borrow the pool; it can be stored in
/// long-lived structs (e.g. a `Task` that outlives the claiming scope). The
/// trade-off: it does not track recycling. Dereferencing after the slot has
/// been [`Fallback::put`] is UB — that contract lives on
/// [`as_ref`](Self::as_ref) / [`as_mut`](Self::as_mut).
///
/// There is no `Drop`: dropping a `HiveOwned` is a no-op (the slot stays
/// claimed). Recycling is the explicit [`Fallback::put`] step, matching the
/// existing pool protocol.
pub struct HiveOwned<T>(NonNull<T>);

impl<T> HiveOwned<T> {
    /// Stable address of the slot. Safe — returning a raw pointer is safe;
    /// dereferencing it isn't.
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }

    /// Borrow the slot's contents for the lifetime of `&self`.
    ///
    /// # Safety
    /// The slot must be fully initialized and must not have been returned to
    /// the pool via [`Fallback::put`] / [`Fallback::put_raw`].
    #[inline]
    pub unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller contract — slot is initialized and not recycled.
        unsafe { self.0.as_ref() }
    }

    /// Mutably borrow the slot's contents for the lifetime of `&mut self`.
    ///
    /// # Safety
    /// The slot must be fully initialized and must not have been returned to
    /// the pool via [`Fallback::put`] / [`Fallback::put_raw`]. No other
    /// reference into the slot may be live for the duration of the borrow.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        // SAFETY: caller contract — slot is initialized, not recycled, and
        // not aliased. `&mut self` prevents forming two `&mut T` from the
        // same handle.
        unsafe { self.0.as_mut() }
    }

    /// Re-seal a pool-slot pointer that round-tripped through an intrusive
    /// queue (`bun_threading::UnboundedQueue`). The queue stores raw `*mut`
    /// links inside the element, so the `HiveOwned` token cannot survive
    /// transit; `push_owned` consumes it, and the drain side reconstructs it
    /// here.
    ///
    /// `#[doc(hidden)]` — this is the queue's private back-door, not general
    /// API. It must have exactly one caller per push/pop boundary; everything
    /// else should hold a real `HiveOwned` from [`Fallback::get_owned`] /
    /// [`HiveSlot::into_owned`].
    ///
    /// # Safety
    /// `ptr` must be the unique owner of a claimed pool slot — i.e. it was
    /// previously the inner pointer of a [`HiveOwned`] that was relinquished
    /// to an intrusive queue (or otherwise not yet `put()` back), and no other
    /// live `HiveOwned` references the same slot.
    #[doc(hidden)]
    #[inline]
    pub unsafe fn from_raw_for_queue(ptr: *mut T) -> Self {
        Self(NonNull::new(ptr).expect("queue node is non-null"))
    }
}

// PORT NOTE: In Zig this was the nested type `HiveArray(T, capacity).Fallback`.
// Rust cannot nest a generic struct that captures outer generics, so it lives at
// module scope with the same parameters. The Zig field
// `hive: if (capacity > 0) Self else void` is always materialized here; the
// `CAPACITY > 0` checks below preserve the original gating.
// PERF(port): zero-capacity case carried a zero-size hive in Zig.
pub struct Fallback<T, const CAPACITY: usize> {
    pub hive: HiveArray<T, CAPACITY>,
    // PORT NOTE: `std.mem.Allocator param` dropped — global mimalloc.
}

impl<T, const CAPACITY: usize> Fallback<T, CAPACITY> {
    pub const fn init() -> Self {
        Self {
            hive: HiveArray::init(),
        }
    }

    /// Placement-new constructor — see [`HiveArray::init_in_place`]. Only
    /// writes the 256 B occupancy bitset; the `[MaybeUninit<T>; CAPACITY]`
    /// buffer is left untouched.
    ///
    /// # Safety
    /// `out` must be non-null, properly aligned, and valid for writes of
    /// `size_of::<Self>()` bytes. The previous contents are not dropped.
    #[inline]
    pub unsafe fn init_in_place(out: *mut Self) {
        // SAFETY: caller contract.
        unsafe { HiveArray::<T, CAPACITY>::init_in_place(core::ptr::addr_of_mut!((*out).hive)) };
    }

    /// Heap-allocate an empty `Fallback` without materializing it on the
    /// stack first.
    ///
    /// `Box::new(Self::init())` is the obvious spelling, but Rust has no
    /// guaranteed result-location semantics: for the 2048-slot
    /// `RequestContext` pool (`sizeof ≈ 816 KB`) LLVM emits the bitset
    /// zeros into a stack temporary and then `memcpy`s the **full** 816 KB
    /// into the heap allocation, committing both ~812 KB of stack pages and
    /// ~812 KB of heap pages that are never read. This entry point allocates
    /// raw heap storage and writes only the 256-byte `used` bitset via
    /// [`init_in_place`](Self::init_in_place); the `[MaybeUninit<T>; CAPACITY]`
    /// buffer is left untouched (uninitialized bytes are a valid bit-pattern
    /// for `MaybeUninit`).
    ///
    /// The returned allocation is leaked — callers stash it in a per-thread
    /// static for the process lifetime (Zig: `threadlocal var pool`).
    #[inline]
    pub fn new_boxed() -> NonNull<Self> {
        let mut boxed = Box::<Self>::new_uninit();
        // SAFETY: `boxed` is a fresh heap allocation — non-null, aligned for
        // `Self`, and valid for writes of `size_of::<Self>()` bytes.
        unsafe { Self::init_in_place(boxed.as_mut_ptr()) };
        // SAFETY: `init_in_place` fully initialized `hive.used`; `hive.buffer`
        // is `[MaybeUninit<T>; CAPACITY]`, for which uninitialized bytes are a
        // valid representation. Every field of `Self` is therefore valid.
        NonNull::from(Box::leak(unsafe { boxed.assume_init() }))
    }

    /// One-shot claim + write. Preferred entry point — no uninit window.
    /// Infallible: spills to a heap `Box<T>` when the inline hive is full.
    #[inline]
    pub fn get_init(&mut self, value: T) -> NonNull<T> {
        self.claim().write(value)
    }

    /// Claim, write, and return a lifetime-erased owned handle. The slot stays
    /// claimed; recycle later via [`put`](Self::put). See [`HiveOwned`].
    #[inline]
    pub fn get_owned(&mut self, value: T) -> HiveOwned<T> {
        HiveOwned(self.get_init(value))
    }

    /// See [`HiveArray::emplace`]. Infallible (heap fallback).
    #[inline]
    pub fn emplace(&mut self, init: impl FnOnce(NonNull<T>) -> T) -> NonNull<T> {
        let slot = self.claim();
        let addr = slot.addr();
        slot.write(init(addr))
    }

    /// See [`HiveArray::claim`]. Infallible: when the inline hive is full,
    /// the returned token owns a freshly-allocated heap slot (tagged so its
    /// `Drop` deallocates without running `T::drop`).
    pub fn claim(&mut self) -> HiveSlot<'_, T, CAPACITY> {
        if CAPACITY > 0 {
            if let Some(slot) = self.hive.claim() {
                return slot;
            }
        }
        let slot = NonNull::from(Box::leak(Box::<T>::new_uninit()));
        HiveSlot {
            slot,
            // Low bit 1 ⇒ heap slot. The hive pointer is not needed on the
            // release path (dealloc is `Box::from_raw(slot)`).
            owner: 1,
            _marker: PhantomData,
        }
    }

    /// Recycle a slot **without** running `T::drop`. Counterpart to
    /// [`HiveArray::put_raw`] for the heap-fallback path.
    ///
    /// # Safety
    /// `value` must have been obtained from this `Fallback` (via `get_init` /
    /// `emplace` / `claim().write()`) and not yet returned. The contained `T`
    /// is **not** dropped — caller must have already moved out / destructured
    /// anything with drop glue, or `T` must be POD.
    pub unsafe fn put_raw(&mut self, value: *mut T) {
        if CAPACITY > 0 {
            if self.hive.put_raw(value) {
                return;
            }
        }
        // SAFETY: caller contract — `value` is a heap slot from `claim()`; it
        // was allocated as `Box<MaybeUninit<T>>` (same layout as `Box<T>`).
        // Reclaiming as `MaybeUninit<T>` deallocates without running
        // `T::drop`.
        drop(unsafe { Box::from_raw(value.cast::<MaybeUninit<T>>()) });
    }

    pub fn r#in(&self, value: *const T) -> bool {
        if CAPACITY > 0 {
            if self.hive.r#in(value) {
                return true;
            }
        }

        false
    }

    /// Return a slot to the pool, dropping the contained `T`.
    ///
    /// # Safety
    /// `value` must point to a fully-initialized `T` previously obtained from
    /// [`get_init`](Self::get_init), [`emplace`](Self::emplace), or from
    /// [`claim`](Self::claim) after consuming the [`HiveSlot`] via
    /// [`HiveSlot::write`] / [`HiveSlot::assume_init`], on this `Fallback`
    /// and not yet returned. A claimed-but-uninitialized slot must not be
    /// passed here.
    pub unsafe fn put(&mut self, value: *mut T) {
        if CAPACITY > 0 {
            // SAFETY: caller contract — `value` is fully initialized.
            if unsafe { self.hive.put(value) } {
                return;
            }
        }

        // SAFETY: `value` was produced by the heap-spill path in `claim()`
        // (it is not in the inline hive) and the caller has since fully
        // initialized it. `destroy` reconstructs the `Box<T>` and runs
        // `T::drop`.
        unsafe { bun_core::heap::destroy(value) };
    }
}

// ──────────────────────────────────────────────────────────────────────────
// HiveRef
// ──────────────────────────────────────────────────────────────────────────
//
// PORT NOTE: ground truth is `bun.HiveRef` in src/bun.zig. It lives here (not
// in the `bun` crate) because every consumer names it through
// `bun_collections::HiveRef`, and its only collaborator is `Fallback` above.
//
// Zig defines `const HiveAllocator = HiveArray(@This(), capacity).Fallback`
// inside the returned struct; Rust spells the self-referential pool type out
// as `Fallback<HiveRef<T, CAPACITY>, CAPACITY>`. CAPACITY is `usize` (widened
// from Zig's `u16`) to line up with `HiveArray`/`Fallback`'s const generic.

/// Intrusive ref-counted slot allocated from a `HiveArray::Fallback` pool.
/// `pool` is a BACKREF (LIFETIMES.tsv class) — the pool strictly outlives
/// every `HiveRef` it hands out, so a raw pointer is the honest mapping.
#[repr(C)]
pub struct HiveRef<T, const CAPACITY: usize> {
    pub ref_count: u32,
    pub pool: *mut Fallback<HiveRef<T, CAPACITY>, CAPACITY>,
    pub value: T,
}

/// Convenience alias mirroring Zig's nested `const HiveAllocator`.
pub type HiveAllocator<T, const CAPACITY: usize> = Fallback<HiveRef<T, CAPACITY>, CAPACITY>;

impl<T, const CAPACITY: usize> HiveRef<T, CAPACITY> {
    /// Zig: `pub fn init(value, allocator) !*@This()`.
    ///
    /// # Safety
    /// `pool` must be valid for the entire lifetime of the returned
    /// `HiveRef` (i.e. until its `ref_count` drops to zero and it is `put`
    /// back). Callers hold the pool in a long-lived owner (e.g. `VirtualMachine`).
    pub unsafe fn init(value: T, pool: *mut Fallback<Self, CAPACITY>) -> *mut Self {
        // SAFETY: caller contract — `pool` is dereferenceable.
        unsafe {
            (*pool)
                .get_init(HiveRef {
                    ref_count: 1,
                    pool,
                    value,
                })
                .as_ptr()
        }
    }

    pub fn ref_(&mut self) -> &mut Self {
        self.ref_count += 1;
        self
    }

    /// Zig: `pub fn unref(this) ?*@This()` — returns `null` when the count hit
    /// zero and the slot was returned to the pool.
    pub fn unref(&mut self) -> Option<&mut Self> {
        let ref_count = self.ref_count;
        self.ref_count = ref_count - 1;
        if ref_count == 1 {
            let pool = self.pool;
            // SAFETY: `self` was produced by `init` above, so `pool` is the
            // pool that owns this slot and is still live (caller contract on
            // `init`). Zig's `if @hasDecl(T, "deinit") this.value.deinit()` maps
            // to `T::drop`, which `Fallback::put` now runs (it drops the whole
            // `HiveRef` in place before recycling/freeing the slot).
            unsafe {
                (*pool).put(std::ptr::from_mut::<Self>(self));
            }
            return None;
        }
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hive_array() {
        const SIZE: usize = 64;

        // Choose an integer with a weird alignment
        // PORT NOTE: Zig used `u127`; Rust has no arbitrary-width ints. `u128` is the closest.
        type Int = u128;

        let mut a = HiveArray::<Int, SIZE>::init();

        {
            let b = a.get_init(0).unwrap().as_ptr();
            assert!(a.get_init(0).unwrap().as_ptr() != b);
            assert_eq!(a.index_of(b), Some(0));
            // SAFETY: `b` is a fully-initialized hive slot.
            assert!(unsafe { a.put(b) });
            assert!(a.get_init(0).unwrap().as_ptr() == b);
            let _c = a.get_init(123).unwrap().as_ptr();
            let mut d: Int = 12345;
            // SAFETY: `&mut d` is foreign — `put` returns `false` and drops nothing.
            assert!(unsafe { a.put(&mut d) } == false);
            assert!(a.r#in(&d) == false);
        }

        // `Int` has no drop glue, so resetting the bitset directly is a safe
        // way to release every slot at once without `put`-ing them one by one.
        a.used = HiveBitSet::init_empty();
        {
            for i in 0..SIZE {
                let b = a.get_init(0).unwrap().as_ptr();
                assert_eq!(a.index_of(b), Some(u32::try_from(i).expect("int cast")));
                // SAFETY: `b` is a fully-initialized hive slot.
                assert!(unsafe { a.put(b) });
                assert!(a.get_init(0).unwrap().as_ptr() == b);
            }
            for _ in 0..SIZE {
                assert!(a.claim().is_none());
            }
        }
    }

    #[test]
    fn hive_slot_drop_releases_without_dtor() {
        use core::sync::atomic::{AtomicU32, Ordering};
        static DROPS: AtomicU32 = AtomicU32::new(0);
        struct D(#[allow(dead_code)] u64);
        impl Drop for D {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut a = HiveArray::<D, 4>::init();
        // Dropped token releases the slot without running D::drop.
        drop(a.claim().unwrap());
        assert!(!a.used.is_set(0));
        assert_eq!(DROPS.load(Ordering::Relaxed), 0);

        // write() commits and put() drops.
        let p = a.get_init(D(7)).unwrap();
        assert!(a.used.is_set(0));
        assert_eq!(DROPS.load(Ordering::Relaxed), 0);
        // SAFETY: `p` is a fully-initialized hive slot.
        unsafe { a.put(p.as_ptr()) };
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);

        // put_raw() does not drop.
        let p = a.get_init(D(8)).unwrap();
        assert!(a.put_raw(p.as_ptr()));
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);

        // Fallback heap path: dropped token deallocates without D::drop.
        let mut f = Fallback::<D, 0>::init();
        drop(f.claim());
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);
        let p = f.get_init(D(9));
        // SAFETY: heap slot from this Fallback.
        unsafe { f.put(p.as_ptr()) };
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn hive_owned_lifecycle() {
        let mut f = Fallback::<u64, 4>::init();

        // get_owned: claim + write + lifetime-erase in one shot.
        let mut owned = f.get_owned(7u64);
        assert!(f.hive.used.is_set(0));
        // SAFETY: `get_owned` initialized the slot; not yet recycled.
        assert_eq!(unsafe { *owned.as_ref() }, 7);
        // SAFETY: same slot, sole `HiveOwned`, not recycled.
        unsafe { *owned.as_mut() = 9 };

        // Move it (compile-time check that it's not implicitly `Copy`).
        let moved = owned;
        // SAFETY: still initialized, slot still claimed.
        assert_eq!(unsafe { *moved.as_ref() }, 9);

        // No `Drop` on `HiveOwned` — recycle through the explicit put().
        // SAFETY: slot is fully initialized; `as_ptr()` is the pool address.
        unsafe { f.put(moved.as_ptr()) };
        assert!(!f.hive.used.is_set(0));

        // claim → into_owned → placement-init → as_ref: the deferred-init path.
        let slot = f.claim();
        // SAFETY: slot will be initialized through `as_ptr()` before any deref.
        let owned = unsafe { slot.into_owned() };
        assert!(f.hive.used.is_set(0));
        // SAFETY: `owned` points at a freshly claimed inline slot.
        unsafe { owned.as_ptr().write(42) };
        // SAFETY: just initialized, slot still claimed.
        assert_eq!(unsafe { *owned.as_ref() }, 42);
        // SAFETY: slot is fully initialized.
        unsafe { f.put(owned.as_ptr()) };
        assert!(!f.hive.used.is_set(0));
    }
}

// ported from: src/collections/hive_array.zig
