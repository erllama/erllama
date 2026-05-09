%% @doc
%% Disk tier server (read_write mode).
%%
%% One server per disk root directory. Owns no ETS tables; the disk
%% itself is the source of truth for slabs. Reads, writes, and deletes
%% all funnel through the gen_server, which keeps file ordering
%% sequential per directory.
%%
%% Save pipeline (`save/3`):
%%
%%   1. Build framed bytes via `erllama_cache_kvc:build/2`.
%%   2. Open writer-unique temp file `<hex>.kvc.<writer_id>.tmp`
%%      with O_EXCL.
%%   3. `prim_file:write/2` an iolist of `[Prefix, Payload]`. Multi-GB
%%      payloads are not concatenated in BEAM memory; the IO subsystem
%%      uses writev under the hood.
%%   4. `prim_file:datasync/1`, then close.
%%   5. `prim_file:make_link/2` to publish at `<hex>.kvc`. EEXIST is
%%      handled: if the existing file is a valid KVC for this Key, we
%%      adopt it (delete our temp). Otherwise we delete the corrupt
%%      file and retry once.
%%   6. `erllama_nif:fsync_dir/1` on the root.
%%   7. Reopen the published file and parse it (validation belt-and-
%%      braces against FS bugs).
%%
%% Returns `{ok, Header, Size}` on success. The caller is responsible
%% for the `meta_srv:reserve_save -> check_reservation -> mark_published
%% -> announce_saved` protocol around this call (the writer pool in
%% step 10 wires that up).
%%
%% On startup, the server scans its directory: deletes any `*.tmp`
%% files (interrupted writes; safe to drop), parses every `<hex>.kvc`
%% header, and registers each valid file with the meta server. Files
%% that fail to parse are deleted.
%% @end
-module(erllama_cache_disk_srv).
-behaviour(gen_server).

-include("erllama_cache.hrl").
-include_lib("kernel/include/file.hrl").

-export([
    start_link/2,
    start_link/3,
    start_link/4,
    save/3,
    load/2,
    delete/2,
    dir/1,
    scan/1
]).

-export([init/1, handle_call/3, handle_cast/2]).

-record(state, {
    name :: atom(),
    tier :: disk | ram_file,
    root :: file:name(),
    disk_io :: read_write | iommap
}).

-type state() :: #state{}.

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link(atom(), file:name()) -> {ok, pid()} | {error, term()}.
start_link(Name, RootDir) ->
    start_link(Name, disk, RootDir, read_write).

-spec start_link(atom(), disk | ram_file, file:name()) ->
    {ok, pid()} | {error, term()}.
start_link(Name, Tier, RootDir) ->
    start_link(Name, Tier, RootDir, read_write).

-spec start_link(atom(), disk | ram_file, file:name(), read_write | iommap) ->
    {ok, pid()} | {error, term()}.
start_link(Name, Tier, RootDir, DiskIO) when
    Tier =:= disk; Tier =:= ram_file
->
    case DiskIO of
        read_write -> ok;
        iommap -> ok
    end,
    gen_server:start_link(
        {local, Name}, ?MODULE, [Name, Tier, RootDir, DiskIO], []
    ).

-spec save(atom(), erllama_cache_kvc:build_meta(), binary()) ->
    {ok, erllama_cache:cache_key(), binary(), non_neg_integer()}
    | {error, term()}.
save(SrvName, BuildMeta, Payload) ->
    gen_server:call(SrvName, {save, BuildMeta, Payload}, infinity).

-spec load(atom(), erllama_cache:cache_key()) ->
    {ok, erllama_cache_kvc:info(), binary()} | miss | {error, term()}.
load(SrvName, Key) ->
    gen_server:call(SrvName, {load, Key}, infinity).

-spec delete(atom(), erllama_cache:cache_key()) -> ok.
delete(SrvName, Key) ->
    gen_server:call(SrvName, {delete, Key}).

-spec dir(atom()) -> file:name().
dir(SrvName) ->
    gen_server:call(SrvName, dir).

%% Scan the directory and return a list of `{Key, Header, Size}` for
%% every parseable `<hex>.kvc`. Side effect: deletes `*.tmp` files
%% and any unreadable `<hex>.kvc` files.
-spec scan(atom()) ->
    [{erllama_cache:cache_key(), binary(), non_neg_integer()}].
scan(SrvName) ->
    gen_server:call(SrvName, scan, infinity).

%% =============================================================================
%% gen_server callbacks
%% =============================================================================

-spec init([term()]) -> {ok, state()}.
init([Name, Tier, Root, DiskIO]) ->
    case filelib:ensure_path(Root) of
        ok -> ok;
        {error, Reason} -> erlang:error({cannot_create_dir, Root, Reason})
    end,
    %% Drop any leftover `*.tmp` from a previous run.
    sweep_tmps(Root),
    %% Register every valid .kvc with the meta server.
    register_existing(Tier, Root),
    {ok, #state{name = Name, tier = Tier, root = Root, disk_io = DiskIO}}.

handle_call({save, BuildMeta, Payload}, _From, S) ->
    {reply, do_save(BuildMeta, Payload, S#state.root), S};
handle_call({load, Key}, _From, S) ->
    {reply, do_load(Key, S#state.root, S#state.disk_io), S};
handle_call({delete, Key}, _From, S) ->
    {reply, do_delete(Key, S#state.root), S};
handle_call(dir, _From, S) ->
    {reply, S#state.root, S};
handle_call(scan, _From, S) ->
    sweep_tmps(S#state.root),
    {reply, scan_dir(S#state.root), S};
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

-spec handle_cast(term(), state()) -> {noreply, state()}.
handle_cast(_Msg, S) ->
    {noreply, S}.

%% =============================================================================
%% Internal: save
%% =============================================================================

do_save(BuildMeta, Payload, Root) ->
    Key = erllama_cache_key:make(#{
        fingerprint => maps:get(fingerprint, BuildMeta),
        quant_type => maps:get(quant_type, BuildMeta),
        ctx_params_hash => maps:get(ctx_params_hash, BuildMeta),
        tokens => maps:get(tokens, BuildMeta)
    }),
    HexKey = bin_to_hex(Key),
    FinalPath = filename:join(Root, HexKey ++ ".kvc"),
    TmpPath = filename:join(Root, HexKey ++ ".kvc." ++ writer_suffix() ++ ".tmp"),
    case erllama_cache_kvc:build(BuildMeta, Payload) of
        {ok, Prefix} ->
            write_then_publish(Key, Prefix, Payload, TmpPath, FinalPath, Root);
        {error, _} = E ->
            E
    end.

write_then_publish(Key, Prefix, Payload, TmpPath, FinalPath, Root) ->
    case write_temp(TmpPath, Prefix, Payload) of
        ok ->
            link_temp(Key, Prefix, Payload, TmpPath, FinalPath, Root);
        {error, _} = E ->
            _ = file:delete(TmpPath),
            E
    end.

write_temp(TmpPath, Prefix, Payload) ->
    case prim_file:open(TmpPath, [write, exclusive, binary, raw]) of
        {ok, Fd} ->
            try
                case prim_file:write(Fd, [Prefix, Payload]) of
                    ok ->
                        prim_file:datasync(Fd);
                    {error, _} = E ->
                        E
                end
            after
                prim_file:close(Fd)
            end;
        {error, _} = E ->
            E
    end.

link_temp(Key, Prefix, Payload, TmpPath, FinalPath, Root) ->
    case try_link(TmpPath, FinalPath) of
        ok ->
            finalise(Key, Prefix, Payload, FinalPath, Root);
        eexist ->
            handle_eexist(Key, Prefix, Payload, TmpPath, FinalPath, Root);
        {error, _} = E ->
            E
    end.

handle_eexist(Key, Prefix, Payload, TmpPath, FinalPath, Root) ->
    case validate_at(FinalPath, Key) of
        {ok, _Info, _Payload} ->
            _ = prim_file:delete(TmpPath),
            ok = erllama_nif:fsync_dir(unicode_path(Root)),
            {ok, Key, header_of(Prefix), file_size(FinalPath)};
        {error, _} ->
            _ = prim_file:delete(FinalPath),
            case try_link(TmpPath, FinalPath) of
                ok -> finalise(Key, Prefix, Payload, FinalPath, Root);
                _ -> {error, eexist}
            end
    end.

try_link(TmpPath, FinalPath) ->
    case prim_file:make_link(TmpPath, FinalPath) of
        ok ->
            _ = prim_file:delete(TmpPath),
            ok;
        {error, eexist} ->
            eexist;
        {error, _} = E ->
            _ = prim_file:delete(TmpPath),
            E
    end.

finalise(Key, Prefix, _Payload, FinalPath, Root) ->
    case erllama_nif:fsync_dir(unicode_path(Root)) of
        ok ->
            case validate_at(FinalPath, Key) of
                {ok, _Info, _ParsedPayload} ->
                    {ok, Key, header_of(Prefix), file_size(FinalPath)};
                {error, _} = E ->
                    _ = prim_file:delete(FinalPath),
                    E
            end;
        {error, _} = E ->
            E
    end.

%% =============================================================================
%% Internal: load / delete
%% =============================================================================

do_load(Key, Root, DiskIO) ->
    Path = filename:join(Root, bin_to_hex(Key) ++ ".kvc"),
    case load_bin(Path, DiskIO) of
        {ok, Bin} -> parse_or_drop(Bin, Key, Path);
        miss -> miss;
        {error, _} = E -> E
    end.

parse_or_drop(Bin, Key, Path) ->
    case erllama_cache_kvc:parse(Bin, Key) of
        {ok, Info, Payload} ->
            {ok, Info, Payload};
        {error, R} ->
            %% Corrupt file: drop it so the next request doesn't
            %% repeat the same failed parse.
            _ = prim_file:delete(Path),
            erllama_cache_counters:incr(?C_CORRUPT_FILES),
            {error, R}
    end.

load_bin(Path, read_write) ->
    case file:read_file(Path) of
        {ok, Bin} -> {ok, Bin};
        {error, enoent} -> miss;
        {error, _} = E -> E
    end;
load_bin(Path, iommap) ->
    %% Zero-copy disk -> BEAM via iommap:region_binary/3. The handle
    %% is closed before this function returns; the returned binary is
    %% a refcounted resource that keeps the underlying mapping alive
    %% via iommap's two-resource lifetime, so the BEAM owns the
    %% memory until GC of the (sub-)binaries.
    case iommap:open(Path, read, []) of
        {ok, H} ->
            try
                Size = file_size(Path),
                case iommap:region_binary(H, 0, Size) of
                    {ok, Bin} -> {ok, Bin};
                    {error, _} = E -> E
                end
            after
                iommap:close(H)
            end;
        {error, enoent} ->
            miss;
        {error, _} = E ->
            E
    end.

do_delete(Key, Root) ->
    Path = filename:join(Root, bin_to_hex(Key) ++ ".kvc"),
    case prim_file:delete(Path) of
        ok -> ok;
        {error, enoent} -> ok;
        {error, _} = E -> E
    end.

%% =============================================================================
%% Internal: scan
%% =============================================================================

sweep_tmps(Root) ->
    case file:list_dir(Root) of
        {ok, Entries} ->
            [
                _ = prim_file:delete(filename:join(Root, E))
             || E <- Entries, lists:suffix(".tmp", E)
            ],
            ok;
        {error, _} ->
            ok
    end.

scan_dir(Root) ->
    case file:list_dir(Root) of
        {ok, Entries} ->
            lists:foldl(
                fun(E, Acc) -> scan_entry(filename:join(Root, E), E, Acc) end,
                [],
                Entries
            );
        {error, _} ->
            []
    end.

scan_entry(Path, Name, Acc) ->
    case lists:suffix(".kvc", Name) andalso (not lists:suffix(".tmp", Name)) of
        true -> scan_kvc(Path, Acc);
        false -> Acc
    end.

scan_kvc(Path, Acc) ->
    case file:read_file(Path) of
        {ok, Bin} ->
            case erllama_cache_kvc:parse_meta(Bin) of
                {ok, Info} ->
                    Header = binary:part(Bin, 0, 48),
                    Tokens = maps:get(tokens, Info),
                    Key = erllama_cache_key:make(#{
                        fingerprint => maps:get(fingerprint, Info),
                        quant_type => maps:get(quant_type, Info),
                        ctx_params_hash => maps:get(ctx_params_hash, Info),
                        tokens => Tokens
                    }),
                    TokensBin = erllama_cache_key:encode_tokens(Tokens),
                    [{Key, Header, byte_size(Bin), TokensBin} | Acc];
                {error, _} ->
                    _ = prim_file:delete(Path),
                    Acc
            end;
        {error, _} ->
            Acc
    end.

register_existing(Tier, Root) ->
    Entries = scan_dir(Root),
    lists:foreach(
        fun({Key, Header, Size, TokensBin}) ->
            Path = filename:join(Root, bin_to_hex(Key) ++ ".kvc"),
            erllama_cache_meta_srv:insert_available(
                Key, Tier, Size, Header, {Tier, Path}, TokensBin
            )
        end,
        Entries
    ).

%% =============================================================================
%% Internal: helpers
%% =============================================================================

validate_at(Path, Key) ->
    case file:read_file(Path) of
        {ok, Bin} -> erllama_cache_kvc:parse(Bin, Key);
        {error, _} = E -> E
    end.

header_of(Prefix) ->
    binary:part(Prefix, 0, 48).

file_size(Path) ->
    case file:read_file_info(Path) of
        {ok, #file_info{size = S}} -> S;
        _ -> 0
    end.

bin_to_hex(Bin) ->
    binary_to_list(binary:encode_hex(Bin, lowercase)).

writer_suffix() ->
    pid_to_list(self()) ++ "." ++ integer_to_list(erlang:unique_integer([positive])).

unicode_path(P) when is_binary(P) -> P;
unicode_path(P) when is_list(P) -> unicode:characters_to_binary(P, utf8).
