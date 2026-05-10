%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_disk_srv_iommap_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% Same disk_srv mechanics as the read_write suite, but with
%% disk_io=iommap. Loads use iommap:region_binary/3 for zero-copy
%% disk -> BEAM. Saves still go through prim_file (write-side
%% zero-copy is not achievable; kv_pack already produced a BEAM
%% binary).

with_iommap(Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(io_disk, disk, Dir, iommap),
    try
        Body(Dir)
    after
        catch gen_server:stop(io_disk),
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

%% =============================================================================
%% Round-trip with iommap loads
%% =============================================================================

iommap_load_round_trips_payload_test() ->
    with_iommap(fun(_Dir) ->
        Meta = base_meta([1, 2, 3, 4]),
        Payload = <<"this loads via iommap">>,
        {ok, Key, _Header, _Size} =
            erllama_cache_disk_srv:save(io_disk, Meta, Payload),
        {ok, _Info, P} = erllama_cache_disk_srv:load(io_disk, Key),
        ?assertEqual(Payload, P)
    end).

iommap_load_handles_large_payload_test() ->
    with_iommap(fun(_Dir) ->
        Meta = base_meta(lists:seq(1, 16)),
        Payload = binary:copy(<<"abcdefgh">>, 1024 * 1024),
        ?assertEqual(8 * 1024 * 1024, byte_size(Payload)),
        {ok, Key, _, _} = erllama_cache_disk_srv:save(io_disk, Meta, Payload),
        {ok, _Info, P} = erllama_cache_disk_srv:load(io_disk, Key),
        ?assertEqual(byte_size(Payload), byte_size(P)),
        ?assertEqual(Payload, P)
    end).

iommap_load_miss_returns_miss_test() ->
    with_iommap(fun(_Dir) ->
        Bogus = crypto:hash(sha256, <<"nope">>),
        ?assertEqual(miss, erllama_cache_disk_srv:load(io_disk, Bogus))
    end).

iommap_load_detects_payload_tamper_test() ->
    with_iommap(fun(Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, Key, _, _} =
            erllama_cache_disk_srv:save(io_disk, Meta, <<"original">>),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        {ok, Bin} = file:read_file(Path),
        Size = byte_size(Bin),
        Head = binary:part(Bin, 0, Size - 8),
        ok = file:write_file(Path, <<Head/binary, "tampered">>),
        ?assertMatch(
            {error, {crc_mismatch, _, _}},
            erllama_cache_disk_srv:load(io_disk, Key)
        )
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_cache_disk_srv_iommap_tests_" ++
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
