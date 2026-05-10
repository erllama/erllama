# Crash-safe publish protocol

This is the contract between the writer pool, the meta server, and
the disk tier. It exists because cache files must appear atomically
to readers, must never be partially written, and must clean up
gracefully when a writer crashes mid-save.

## The five-stage protocol

```
┌─ writer process ────────────────────────────────────────────────┐
│                                                                 │
│  1.  reserve_save(Key, Tier, Bytes)                             │
│         meta server inserts a reservation row,                  │
│         returns a fresh ReservationToken.                       │
│                                                                 │
│  2.  write_tmp(Path.tmp, Bytes)                                 │
│         streamed prim_file:write/2; fdatasync at end.           │
│                                                                 │
│  3.  check_reservation(Key, ReservationToken)                   │
│         meta server confirms the reservation is still live      │
│         (no concurrent writer has superseded us).               │
│                                                                 │
│  4.  link(Path.tmp, Path)                                       │
│         atomic create-if-not-exists; the only durable           │
│         publish step. EEXIST is validated and either            │
│         adopted or replaced under the current reservation.      │
│                                                                 │
│  5.  mark_published(Key, ReservationToken)                      │
│         meta server flips the reservation row to a              │
│         published meta row, then announces to subscribers.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why each stage exists

### Stage 1: reserve

Prevents two concurrent writers for the same key from both reaching
stage 4 and racing on `link(2)`. The reservation is a row in the
meta ETS keyed on `Key`, with the writer's pid and a monotonic
token. Subsequent `reserve_save` calls for the same key during the
window block or fail-fast depending on policy.

### Stage 2: write_tmp

The temp file lives next to the final path
(`Path.tmp.<pid>.<token>`) so `link(2)` is always intra-directory
(no cross-fs hops). We `fdatasync` the temp file before stage 3 to
ensure stage 4's atomic publish actually publishes durable bytes —
otherwise a crash between stage 4 and the fs's writeback could
expose a zero-length file as the canonical row.

### Stage 3: check_reservation

The window between stage 1 and stage 4 can be tens of milliseconds
on slow disks. In that window:

- The writer could have been killed and respawned by the supervisor.
  A second reservation for the same key may exist with a fresh
  token.
- An `evict` could have decided this key is dead.

`check_reservation` is the meta server confirming our token is
still the live one before we publish. A negative result means
"someone else is doing this; abandon and clean up".

### Stage 4: link

`link(2)` is the durable publish. It is atomic create-if-not-exists
on POSIX file systems: either the file did not exist and we created
the hardlink, or the file existed and we got `EEXIST`. We never
silently swallow `EEXIST` — we open the existing file and either:

- **Adopt** it if the parsed header matches our reservation token's
  expected key + size. This handles the case where a previous
  writer crashed *after* `link` but *before* `mark_published`; the
  file is good, just unannounced.
- **Replace** it if it parses as a different key (collision, very
  rare) or fails to parse. We unlink and link our temp again.

Adopt-or-replace is the only path that is correct under writer
crashes and orphan files. Skipping `EEXIST` is a footgun;
specifically forbidden in `AGENTS.md`.

### Stage 5: mark_published

The reservation flips to a published meta row in a single
gen_server call. Subscribers (multi-turn waiters from
`session_resume_wait_ms`) are notified by `gen_server:reply/2` to
the parked callers. Reads now find the key on the index.

## Two-stage TTL cleanup

Reservations have a TTL. A writer that crashes between stages 1 and
5 leaves a stale reservation; the meta server reaps them in two
stages:

1. **TTL elapsed, stage ≤ 3:** drop the reservation, no file action
   needed (writer hadn't published yet).
2. **TTL elapsed, stage 4 reached:** check disk. If the linked file
   exists and parses, *adopt* it under a fresh reservation. If it
   fails to parse, unlink and drop.

This is what makes orphan adoption work: a writer can crash
arbitrarily late and we still recover the bytes it wrote.

## What `disk_io = iommap` adds

When the disk tier is in `iommap` mode, reads happen against a
zero-copy refcounted region binary (`iommap:region_binary/3`). The
reservation protocol stays the same; the read path changes.

Cost of `iommap` is that erllama must have **exclusive** access to
the cache directory. An external `truncate(2)` on a published file
would invalidate live region binaries; the BEAM would surface that
as a SIGBUS at message-send or sub-binary creation time. The disk
tier server takes an advisory `flock(LOCK_EX)` on a sentinel file
at startup to detect concurrent erllama processes; it does not
defend against unrelated tools.

The `read_write` mode does not have this problem because every read
copies into a fresh binary. Pick `auto` and trust the platform's
default.

## Test surface

Each invariant has a dedicated case:

| Invariant | Test |
|---|---|
| `EEXIST` adopted when header matches | `erllama_cache_writer_tests:eexist_adopt/0` |
| `EEXIST` replaced when header is junk | `erllama_cache_writer_tests:eexist_replace/0` |
| Stale reservation reaped, file adopted | `erllama_cache_meta_SUITE:ttl_orphan_adopt/0` |
| Stale reservation reaped, file unlinked | `erllama_cache_meta_SUITE:ttl_orphan_drop/0` |
| Concurrent writers: one wins, others abandon | `prop_cache_publish:prop_publish_serialises/0` |
| Multi-turn `parent_key` waits for in-flight finish | `erllama_cache_tests:resume_waits_for_finish/0` |

If you change any stage of the protocol, surface tension with these
tests to a reviewer before landing.
