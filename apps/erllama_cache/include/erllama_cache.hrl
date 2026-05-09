%% erllama_cache: shared records, type aliases, and ETS row position constants.
%%
%% Position constants are referenced from erllama_cache_meta_srv via
%% ets:update_counter / ets:update_element on a flat tuple row.

-ifndef(ERLLAMA_CACHE_HRL).
-define(ERLLAMA_CACHE_HRL, true).

%% Meta row position constants. Layout:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, HeaderBin, Location, TokensRef}
-define(POS_KEY, 1).
-define(POS_TIER, 2).
-define(POS_SIZE, 3).
-define(POS_LAST_USED, 4).
-define(POS_REFCOUNT, 5).
-define(POS_STATUS, 6).
-define(POS_HEADER_BIN, 7).
-define(POS_LOCATION, 8).
-define(POS_TOKENS_REF, 9).

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
-define(C_NSLOTS, 16).

%% Public types live in `erllama_cache.erl`. Refer to them as
%% `erllama_cache:cache_key()` etc. from outside this header.

-endif.
