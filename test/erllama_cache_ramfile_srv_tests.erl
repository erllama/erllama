-module(erllama_cache_ramfile_srv_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% The ram_file tier reuses erllama_cache_disk_srv internals via a
%% thin delegating wrapper. The tests below exercise the wrapper API
%% surface plus the tier-label propagation into the meta server; the
%% bulk of file-tier behaviour is covered by the disk_srv suite.

with_ramfile(Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_ramfile_srv:start_link(test_ramfile, Dir),
    try
        Body(Dir)
    after
        catch gen_server:stop(test_ramfile),
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

save_then_load_round_trip_test() ->
    with_ramfile(fun(_Dir) ->
        Meta = base_meta([1, 2, 3]),
        Payload = <<"ramfile payload">>,
        {ok, Key, _Header, _Size} =
            erllama_cache_ramfile_srv:save(test_ramfile, Meta, Payload),
        ?assertEqual(Key, key_for(Meta)),
        {ok, _Info, P} = erllama_cache_ramfile_srv:load(test_ramfile, Key),
        ?assertEqual(Payload, P)
    end).

init_registers_with_ram_file_tier_label_test() ->
    Dir = make_tmp_dir(),
    try
        Meta = base_meta([1, 2, 3, 4, 5, 6, 7, 8]),
        Key = key_for(Meta),
        {ok, Prefix} = erllama_cache_kvc:build(Meta, <<"x">>),
        Path = filename:join(Dir, bin_to_hex(Key) ++ ".kvc"),
        ok = file:write_file(Path, <<Prefix/binary, "x">>),
        {ok, _} = erllama_cache_meta_srv:start_link(),
        {ok, _} = erllama_cache_ram:start_link(),
        {ok, _} = erllama_cache_ramfile_srv:start_link(scan_ramfile, Dir),
        try
            {ok, Row} = erllama_cache_meta_srv:lookup_exact(Key),
            ?assertEqual(ram_file, element(?POS_TIER, Row)),
            ?assertMatch({ram_file, _}, element(?POS_LOCATION, Row))
        after
            catch gen_server:stop(scan_ramfile),
            catch gen_server:stop(erllama_cache_ram),
            catch gen_server:stop(erllama_cache_meta_srv)
        end
    after
        rm_rf(Dir)
    end.

eviction_deletes_ram_file_too_test() ->
    with_ramfile(fun(Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, Key, _, _} =
            erllama_cache_ramfile_srv:save(test_ramfile, Meta, <<"abc">>),
        %% Register manually since save here does not currently route
        %% through the reservation protocol (writer pool wires that up
        %% in step 10). Use insert_available to make the row visible.
        {ok, Bin} = file:read_file(filename:join(Dir, bin_to_hex(Key) ++ ".kvc")),
        Header = binary:part(Bin, 0, 48),
        ok = erllama_cache_meta_srv:insert_available(
            Key,
            ram_file,
            byte_size(Bin),
            Header,
            {ram_file, filename:join(Dir, bin_to_hex(Key) ++ ".kvc")}
        ),
        {evicted, 1} = erllama_cache_meta_srv:gc(),
        ?assertEqual(miss, erllama_cache_ramfile_srv:load(test_ramfile, Key))
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_cache_ramfile_srv_tests_" ++
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
