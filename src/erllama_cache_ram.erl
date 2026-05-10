%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% RAM tier slab store.
%%
%% Owns the `erllama_cache_ram_slabs` ETS table. Reads and deletes
%% are direct ETS operations from any process; puts go through this
%% module's `put/3` because they need to atomically (a) insert the
%% slab and (b) register the meta row via the meta server.
%%
%% Slab binaries above 64 bytes live off-heap as refcounted ProcBins
%% so `ets:lookup` returns a binary reference, not a copy. There is
%% no slab fragmentation to worry about; eviction frees the slab
%% binary by deleting its row.
%% @end
-module(erllama_cache_ram).
-behaviour(gen_server).

-export([start_link/0, put/3, load/1, delete/1, size_bytes/0]).
-export([init/1, handle_call/3, handle_cast/2]).

-define(SLABS, erllama_cache_ram_slabs).
-define(SERVER, ?MODULE).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

-spec put(erllama_cache:cache_key(), binary(), binary()) -> ok.
put(Key, Slab, Header) when is_binary(Slab), is_binary(Header) ->
    ets:insert(?SLABS, {Key, Slab}),
    erllama_cache_meta_srv:insert_available(
        Key, ram, byte_size(Slab), Header, {ram}
    ).

-spec load(erllama_cache:cache_key()) -> {ok, binary()} | miss.
load(Key) ->
    case ets:lookup(?SLABS, Key) of
        [{_, Slab}] -> {ok, Slab};
        [] -> miss
    end.

-spec delete(erllama_cache:cache_key()) -> ok.
delete(Key) ->
    %% Idempotent against a non-existent table; the ram tier may not
    %% have been started yet (early init) or may have crashed and not
    %% yet been restarted.
    try
        ets:delete(?SLABS, Key),
        ok
    catch
        error:badarg -> ok
    end.

-spec size_bytes() -> non_neg_integer().
size_bytes() ->
    ets:foldl(fun({_, S}, Acc) -> Acc + byte_size(S) end, 0, ?SLABS).

%% =============================================================================
%% gen_server callbacks
%% =============================================================================

-spec init([]) -> {ok, []}.
init([]) ->
    ets:new(?SLABS, [
        named_table,
        public,
        set,
        {read_concurrency, true},
        {write_concurrency, true}
    ]),
    {ok, []}.

handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

-spec handle_cast(term(), term()) -> {noreply, term()}.
handle_cast(_Msg, S) ->
    {noreply, S}.
