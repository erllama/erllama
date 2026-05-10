%% @doc
%% Public façade for the cache subsystem.
%%
%% Tier servers, the meta server, the writer pool, and the policy
%% module are internal. The runtime cache plumbing (`lookup_exact/1`,
%% `checkout/2`, `checkin/1`, `save_async/N`, `lookup_exact_or_wait/2`)
%% lives in `erllama_cache_meta_srv` and `erllama_cache_writer`; this
%% module exposes only the operator-friendly subset.
%% @end
-module(erllama_cache).

-export([get_counters/0, reset_counters/0, gc/0, evict_bytes/1, evict_bytes/2]).

-export_type([
    cache_key/0,
    tier/0,
    status/0,
    save_reason/0
]).

%% @doc Snapshot of operational counters as a map of slot name to
%% non-negative integer. Suitable for forwarding to a metrics
%% exporter (Prometheus, statsd, OpenTelemetry).
-spec get_counters() -> #{atom() => non_neg_integer()}.
get_counters() ->
    erllama_cache_counters:snapshot().

%% @doc Reset all counters to 0. Mostly for tests; production
%% callers should treat counters as monotonic-since-boot.
-spec reset_counters() -> ok.
reset_counters() ->
    erllama_cache_counters:reset().

%% @doc Synchronous full eviction pass. Walks the LRU and drops every
%% available row with `refcount = 0`. Returns the number evicted.
-spec gc() -> {evicted, non_neg_integer()}.
gc() ->
    erllama_cache_meta_srv:gc().

%% @doc Evict oldest available rows until at least `TargetBytes` have
%% been freed. Returns `{evicted, NumRows, BytesFreed}`.
-spec evict_bytes(non_neg_integer()) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes) ->
    erllama_cache_meta_srv:evict_bytes(TargetBytes).

%% @doc Like `evict_bytes/1`, but only considers rows whose tier is in
%% `Tiers`. Pass `all` to match every tier, or a subset of
%% `[ram, ram_file, disk]`. The system-pressure scheduler uses this to
%% evict only RAM-resident slabs while leaving the disk tier alone.
-spec evict_bytes(non_neg_integer(), all | [tier()]) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes, Tiers) ->
    erllama_cache_meta_srv:evict_bytes(TargetBytes, Tiers).

-type cache_key() :: <<_:256>>.
-type tier() :: ram | ram_file | disk.
-type status() :: available | writing | evicting.
-type save_reason() :: cold | continued | finish | evict | shutdown.
