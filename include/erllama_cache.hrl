%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% erllama_cache: shared records, type aliases, and ETS row position constants.
%%
%% Position constants are referenced from erllama_cache_meta_srv via
%% ets:update_counter / ets:update_element on a flat tuple row.

-ifndef(ERLLAMA_CACHE_HRL).
-define(ERLLAMA_CACHE_HRL, true).

%% Meta row position constants. Layout:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, HeaderBin, Location,
%%    TokensRef, Hits}
%% Hits is a per-row count of warm reuses. Persisted at offset 12 of
%% the on-disk KVC header so eviction scoring survives restarts.
-define(POS_KEY, 1).
-define(POS_TIER, 2).
-define(POS_SIZE, 3).
-define(POS_LAST_USED, 4).
-define(POS_REFCOUNT, 5).
-define(POS_STATUS, 6).
-define(POS_HEADER_BIN, 7).
-define(POS_LOCATION, 8).
-define(POS_TOKENS_REF, 9).
-define(POS_HITS, 10).

%% Offset of the u32 hit_count inside the 48-byte KVC header.
%% magic(3) + version(1) + quant(1) + reason(1) + reserved(2) +
%% token_count(4) = 12.
-define(KVC_HEADER_HITS_OFFSET, 12).

%% Atomics counter slots. See plans/golden-finding-horizon.md Q10.
-define(C_HITS_EXACT, 1).
-define(C_HITS_RESUME, 2).
-define(C_MISSES, 3).
-define(C_SAVES_COLD, 4).
-define(C_SAVES_CONTINUED, 5).
-define(C_SAVES_FINISH, 6).
-define(C_SAVES_EVICT, 7).
-define(C_SAVES_SHUTDOWN, 8).
-define(C_EVICTIONS, 9).
-define(C_CORRUPT_FILES, 10).
-define(C_PACK_TOTAL_NS, 11).
-define(C_LOAD_TOTAL_NS, 12).
-define(C_BYTES_RAM, 13).
-define(C_BYTES_RAMFILE, 14).
-define(C_BYTES_DISK, 15).
-define(C_DUPLICATE_DROPPED, 16).
%% Longest-prefix walk profiling. probes is the running sum of probes
%% attempted across all walks; probes / (hits_longest_prefix + misses)
%% gives the average walk depth. ns is total monotonic time spent in
%% the walk.
-define(C_HITS_LONGEST_PREFIX, 17).
-define(C_LONGEST_PREFIX_PROBES, 18).
-define(C_LONGEST_PREFIX_NS, 19).
%% Bumped when erllama_cache_writer:save returns {error, max_concurrent}
%% or {error, already_present}. The model layer fires saves as
%% fire-and-forget; this counter is the only surface that reveals a
%% back-pressured writer dropping a save the model wanted to fire.
-define(C_SAVES_DROPPED, 20).
-define(C_NSLOTS, 20).

%% Public types live in `erllama_cache.erl`. Refer to them as
%% `erllama_cache:cache_key()` etc. from outside this header.

-endif.
