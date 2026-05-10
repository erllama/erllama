%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% RAM-file tier server.
%%
%% Identical mechanics to `erllama_cache_disk_srv` (same KVC framing,
%% same temp+datasync+link+fsync_dir publish protocol) — the only
%% difference is the root directory points at a tmpfs mount such as
%% `/dev/shm` so the bytes never touch a spinning disk. The tier
%% label `ram_file` is what the meta server records, which lets the
%% scheduler reason about durability separately from latency.
%%
%% Implementation is a delegating wrapper; all real work lives in
%% `erllama_cache_disk_srv`.
%% @end
-module(erllama_cache_ramfile_srv).

-export([start_link/2, save/3, load/2, delete/2, dir/1, scan/1]).

-spec start_link(atom(), file:name()) -> {ok, pid()} | {error, term()}.
start_link(Name, RootDir) ->
    erllama_cache_disk_srv:start_link(Name, ram_file, RootDir).

-spec save(atom(), erllama_cache_kvc:build_meta(), binary()) ->
    {ok, erllama_cache:cache_key(), binary(), non_neg_integer()}
    | {error, term()}.
save(SrvName, BuildMeta, Payload) ->
    erllama_cache_disk_srv:save(SrvName, BuildMeta, Payload).

-spec load(atom(), erllama_cache:cache_key()) ->
    {ok, erllama_cache_kvc:info(), binary()} | miss | {error, term()}.
load(SrvName, Key) ->
    erllama_cache_disk_srv:load(SrvName, Key).

-spec delete(atom(), erllama_cache:cache_key()) -> ok.
delete(SrvName, Key) ->
    erllama_cache_disk_srv:delete(SrvName, Key).

-spec dir(atom()) -> file:name().
dir(SrvName) ->
    erllama_cache_disk_srv:dir(SrvName).

-spec scan(atom()) ->
    [{erllama_cache:cache_key(), binary(), non_neg_integer()}].
scan(SrvName) ->
    erllama_cache_disk_srv:scan(SrvName).
