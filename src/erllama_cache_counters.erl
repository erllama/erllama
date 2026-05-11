%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_cache_counters).
-moduledoc """
Cache subsystem operational counters.

A single `atomics` array (one slot per metric) lives in
`persistent_term` keyed by this module. Hot paths bump slots via
`incr/1,2`; readers take a snapshot via `snapshot/0`.

Slot indices are defined in `include/erllama_cache.hrl` as `?C_*`
macros so callers can pass the symbolic constant rather than a
magic integer.

Adapter integrations (Prometheus, statsd, OpenTelemetry, etc.) read
`snapshot/0` periodically; this module does not depend on any
external metrics framework.
""".

-include("erllama_cache.hrl").

-export([
    init/0,
    incr/1,
    incr/2,
    add/2,
    get/1,
    snapshot/0,
    reset/0
]).

-define(ARR_KEY, {?MODULE, atomics}).

%% =============================================================================
%% Public API
%% =============================================================================

%% Idempotent: subsequent calls no-op (the atomics array survives
%% as long as persistent_term holds it).
-spec init() -> ok.
init() ->
    case persistent_term:get(?ARR_KEY, undefined) of
        undefined ->
            A = atomics:new(?C_NSLOTS, [{signed, false}]),
            persistent_term:put(?ARR_KEY, A),
            ok;
        _Existing ->
            ok
    end.

-spec incr(pos_integer()) -> ok.
incr(Slot) ->
    add(Slot, 1).

-spec incr(pos_integer(), non_neg_integer()) -> ok.
incr(Slot, N) ->
    add(Slot, N).

-spec add(pos_integer(), non_neg_integer()) -> ok.
add(Slot, N) ->
    case persistent_term:get(?ARR_KEY, undefined) of
        undefined -> ok;
        A -> atomics:add(A, Slot, N)
    end.

-spec get(pos_integer()) -> non_neg_integer().
get(Slot) ->
    case persistent_term:get(?ARR_KEY, undefined) of
        undefined -> 0;
        A -> atomics:get(A, Slot)
    end.

-spec snapshot() -> #{atom() => non_neg_integer()}.
snapshot() ->
    case persistent_term:get(?ARR_KEY, undefined) of
        undefined ->
            #{};
        A ->
            maps:from_list([
                {slot_name(N), atomics:get(A, N)}
             || N <- lists:seq(1, ?C_NSLOTS)
            ])
    end.

-spec reset() -> ok.
reset() ->
    case persistent_term:get(?ARR_KEY, undefined) of
        undefined ->
            ok;
        A ->
            lists:foreach(fun(N) -> atomics:put(A, N, 0) end, lists:seq(1, ?C_NSLOTS))
    end.

%% =============================================================================
%% Internal: slot -> atom name mapping
%% =============================================================================

slot_name(?C_HITS_EXACT) -> hits_exact;
slot_name(?C_HITS_RESUME) -> hits_resume;
slot_name(?C_MISSES) -> misses;
slot_name(?C_SAVES_COLD) -> saves_cold;
slot_name(?C_SAVES_CONTINUED) -> saves_continued;
slot_name(?C_SAVES_FINISH) -> saves_finish;
slot_name(?C_SAVES_EVICT) -> saves_evict;
slot_name(?C_SAVES_SHUTDOWN) -> saves_shutdown;
slot_name(?C_EVICTIONS) -> evictions;
slot_name(?C_CORRUPT_FILES) -> corrupt_files;
slot_name(?C_PACK_TOTAL_NS) -> pack_total_ns;
slot_name(?C_LOAD_TOTAL_NS) -> load_total_ns;
slot_name(?C_BYTES_RAM) -> bytes_ram;
slot_name(?C_BYTES_RAMFILE) -> bytes_ramfile;
slot_name(?C_BYTES_DISK) -> bytes_disk;
slot_name(?C_DUPLICATE_DROPPED) -> duplicate_dropped;
slot_name(?C_HITS_LONGEST_PREFIX) -> hits_longest_prefix;
slot_name(?C_LONGEST_PREFIX_PROBES) -> longest_prefix_probes;
slot_name(?C_LONGEST_PREFIX_NS) -> longest_prefix_ns;
slot_name(?C_SAVES_DROPPED) -> saves_dropped.
