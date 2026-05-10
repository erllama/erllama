%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_disk_srv_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_disk(Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(test_disk, Dir),
    try
        Body(Dir)
    after
        catch gen_server:stop(test_disk),
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv),
        rm_rf(Dir)
    end.

base_meta(Tokens) ->
    #{
        save_reason => cold,
        quant_bits => 16,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => Tokens,
        context_size => 4096,
        creation_time => 1000,
        last_used_time => 1000,
        hit_count => 0,
        prompt_text => <<>>,
        hostname => <<"test">>,
        erllama_version => <<"0.1.0">>
    }.

key_for(Meta) ->
    erllama_cache_key:make(#{
        fingerprint => maps:get(fingerprint, Meta),
        quant_type => maps:get(quant_type, Meta),
        ctx_params_hash => maps:get(ctx_params_hash, Meta),
        tokens => maps:get(tokens, Meta)
    }).

%% =============================================================================
%% Save / load round-trip
%% =============================================================================

save_then_load_round_trip_test() ->
    with_disk(fun(_Dir) ->
        Meta = base_meta([1, 2, 3]),
        Payload = <<"some kv state">>,
        {ok, Key, _Header, Size} = erllama_cache_disk_srv:save(test_disk, Meta, Payload),
        ?assert(Size > byte_size(Payload)),
        ?assertEqual(Key, key_for(Meta)),
        {ok, _Info, P} = erllama_cache_disk_srv:load(test_disk, Key),
        ?assertEqual(Payload, P)
    end).

save_writes_only_one_file_test() ->
    with_disk(fun(Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, _Key, _Header, _Size} =
            erllama_cache_disk_srv:save(test_disk, Meta, <<"x">>),
        {ok, Entries} = file:list_dir(Dir),
        Kvc = [E || E <- Entries, lists:suffix(".kvc", E)],
        Tmp = [E || E <- Entries, lists:suffix(".tmp", E)],
        ?assertEqual(1, length(Kvc)),
        ?assertEqual([], Tmp)
    end).

load_unknown_key_returns_miss_test() ->
    with_disk(fun(_Dir) ->
        Bogus = crypto:hash(sha256, <<"nope">>),
        ?assertEqual(miss, erllama_cache_disk_srv:load(test_disk, Bogus))
    end).

%% =============================================================================
%% Delete
%% =============================================================================

delete_removes_file_test() ->
    with_disk(fun(_Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, Key, _, _} = erllama_cache_disk_srv:save(test_disk, Meta, <<"x">>),
        ok = erllama_cache_disk_srv:delete(test_disk, Key),
        ?assertEqual(miss, erllama_cache_disk_srv:load(test_disk, Key))
    end).

delete_unknown_key_is_idempotent_test() ->
    with_disk(fun(_Dir) ->
        Bogus = crypto:hash(sha256, <<"nope">>),
        ?assertEqual(ok, erllama_cache_disk_srv:delete(test_disk, Bogus))
    end).

%% =============================================================================
%% Corruption detection
%% =============================================================================

load_detects_payload_tamper_test() ->
    with_disk(fun(Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, Key, _, _} =
            erllama_cache_disk_srv:save(test_disk, Meta, <<"original">>),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        %% Tamper the last 8 bytes (the payload).
        {ok, Bin} = file:read_file(Path),
        Size = byte_size(Bin),
        Head = binary:part(Bin, 0, Size - 8),
        Tampered = <<Head/binary, "tampered">>,
        ok = file:write_file(Path, Tampered),
        ?assertMatch(
            {error, {crc_mismatch, _, _}},
            erllama_cache_disk_srv:load(test_disk, Key)
        ),
        %% After detection the file is deleted (so the same request
        %% doesn't repeat the parse on every retry).
        ?assertEqual(miss, erllama_cache_disk_srv:load(test_disk, Key))
    end).

%% =============================================================================
%% On-start scan
%% =============================================================================

init_sweeps_tmp_files_test() ->
    Dir = make_tmp_dir(),
    try
        ok = file:write_file(filename:join(Dir, "abc.kvc.tmp"), <<"junk">>),
        ok = file:write_file(filename:join(Dir, "def.kvc.foo.tmp"), <<"junk">>),
        {ok, _} = erllama_cache_meta_srv:start_link(),
        {ok, _} = erllama_cache_ram:start_link(),
        {ok, _} = erllama_cache_disk_srv:start_link(scan_disk, Dir),
        try
            {ok, Entries} = file:list_dir(Dir),
            ?assertEqual([], Entries)
        after
            catch gen_server:stop(scan_disk),
            catch gen_server:stop(erllama_cache_ram),
            catch gen_server:stop(erllama_cache_meta_srv)
        end
    after
        rm_rf(Dir)
    end.

init_registers_existing_kvc_files_test() ->
    Dir = make_tmp_dir(),
    try
        Meta = base_meta([1, 2, 3, 4, 5, 6, 7, 8]),
        Key = key_for(Meta),
        {ok, Prefix} = erllama_cache_kvc:build(Meta, <<"payload">>),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        ok = file:write_file(Path, <<Prefix/binary, "payload">>),
        {ok, _} = erllama_cache_meta_srv:start_link(),
        {ok, _} = erllama_cache_ram:start_link(),
        {ok, _} = erllama_cache_disk_srv:start_link(scan2_disk, Dir),
        try
            ?assertMatch({ok, _Row}, erllama_cache_meta_srv:lookup_exact(Key))
        after
            catch gen_server:stop(scan2_disk),
            catch gen_server:stop(erllama_cache_ram),
            catch gen_server:stop(erllama_cache_meta_srv)
        end
    after
        rm_rf(Dir)
    end.

init_drops_corrupt_kvc_files_test() ->
    Dir = make_tmp_dir(),
    try
        ok = file:write_file(filename:join(Dir, "deadbeef.kvc"), <<"not a kvc">>),
        {ok, _} = erllama_cache_meta_srv:start_link(),
        {ok, _} = erllama_cache_ram:start_link(),
        {ok, _} = erllama_cache_disk_srv:start_link(scan3_disk, Dir),
        try
            {ok, Entries} = file:list_dir(Dir),
            ?assertEqual([], Entries)
        after
            catch gen_server:stop(scan3_disk),
            catch gen_server:stop(erllama_cache_ram),
            catch gen_server:stop(erllama_cache_meta_srv)
        end
    after
        rm_rf(Dir)
    end.

%% =============================================================================
%% EEXIST adopt / replace
%% =============================================================================

eexist_with_valid_existing_file_adopts_test() ->
    with_disk(fun(Dir) ->
        %% Write a valid file with the same key out-of-band, then save again.
        Meta = base_meta([1, 2, 3]),
        Key = key_for(Meta),
        {ok, Prefix} = erllama_cache_kvc:build(Meta, <<"first">>),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        ok = file:write_file(Path, <<Prefix/binary, "first">>),
        %% Now save a different payload for the same logical state.
        %% (Both have identical Key by construction.)
        {ok, AdoptedKey, _Header, _Size} =
            erllama_cache_disk_srv:save(test_disk, Meta, <<"second">>),
        ?assertEqual(Key, AdoptedKey),
        %% The first file's content wins.
        {ok, _Info, P} = erllama_cache_disk_srv:load(test_disk, Key),
        ?assertEqual(<<"first">>, P)
    end).

eexist_with_corrupt_existing_file_replaces_test() ->
    with_disk(fun(Dir) ->
        Meta = base_meta([1, 2, 3]),
        Key = key_for(Meta),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        ok = file:write_file(Path, <<"corrupt">>),
        {ok, _Key, _Header, _Size} =
            erllama_cache_disk_srv:save(test_disk, Meta, <<"good payload">>),
        {ok, _Info, P} = erllama_cache_disk_srv:load(test_disk, Key),
        ?assertEqual(<<"good payload">>, P)
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_cache_disk_srv_tests_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:make_dir(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} -> [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ -> ok
    end,
    file:del_dir(Dir).

bin_to_hex(Bin) ->
    binary_to_list(binary:encode_hex(Bin, lowercase)).
