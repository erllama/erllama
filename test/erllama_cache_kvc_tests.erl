-module(erllama_cache_kvc_tests).
-include_lib("eunit/include/eunit.hrl").

fp() -> binary:copy(<<16#AA>>, 32).
ctx_hash() -> binary:copy(<<16#BB>>, 32).

base_meta() ->
    #{
        save_reason => cold,
        quant_bits => 16,
        fingerprint => fp(),
        fingerprint_mode => safe,
        quant_type => f16,
        ctx_params_hash => ctx_hash(),
        tokens => [1, 2, 3, 4, 5],
        context_size => 4096,
        creation_time => 1234567890,
        last_used_time => 1234567990,
        hit_count => 0,
        prompt_text => <<"hello">>,
        hostname => <<"testhost">>,
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
%% Round-trip
%% =============================================================================

build_then_parse_round_trip_test() ->
    Meta = base_meta(),
    Payload = <<"this is the kv state payload">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary>>,
    Key = key_for(Meta),
    {ok, Info, ParsedPayload} = erllama_cache_kvc:parse(Bin, Key),
    ?assertEqual(Payload, ParsedPayload),
    ?assertEqual(cold, maps:get(save_reason, Info)),
    ?assertEqual(16, maps:get(quant_bits, Info)),
    ?assertEqual(fp(), maps:get(fingerprint, Info)),
    ?assertEqual(safe, maps:get(fingerprint_mode, Info)),
    ?assertEqual(f16, maps:get(quant_type, Info)),
    ?assertEqual(ctx_hash(), maps:get(ctx_params_hash, Info)),
    ?assertEqual([1, 2, 3, 4, 5], maps:get(tokens, Info)),
    ?assertEqual(<<"hello">>, maps:get(prompt_text, Info)),
    ?assertEqual(<<"testhost">>, maps:get(hostname, Info)),
    ?assertEqual(<<"0.1.0">>, maps:get(erllama_version, Info)).

build_with_empty_payload_round_trip_test() ->
    Meta = base_meta(),
    Payload = <<>>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary>>,
    Key = key_for(Meta),
    {ok, _Info, <<>>} = erllama_cache_kvc:parse(Bin, Key).

build_with_large_payload_round_trip_test() ->
    Meta = base_meta(),
    Payload = binary:copy(<<"abc">>, 1_000_000),
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary>>,
    Key = key_for(Meta),
    {ok, _Info, ParsedPayload} = erllama_cache_kvc:parse(Bin, Key),
    ?assertEqual(byte_size(Payload), byte_size(ParsedPayload)),
    ?assertEqual(Payload, ParsedPayload).

build_with_empty_prompt_text_test() ->
    Meta = (base_meta())#{prompt_text => <<>>},
    Payload = <<"x">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary>>,
    Key = key_for(Meta),
    {ok, Info, _} = erllama_cache_kvc:parse(Bin, Key),
    ?assertEqual(<<>>, maps:get(prompt_text, Info)).

%% =============================================================================
%% Header layout (ds4-compatible) sanity checks
%% =============================================================================

header_starts_with_magic_test() ->
    {ok, Prefix} = erllama_cache_kvc:build(base_meta(), <<"x">>),
    <<Magic:3/binary, _/binary>> = Prefix,
    ?assertEqual(<<"KVC">>, Magic).

header_version_is_one_test() ->
    {ok, Prefix} = erllama_cache_kvc:build(base_meta(), <<"x">>),
    <<_:3/binary, Version:8, _/binary>> = Prefix,
    ?assertEqual(1, Version).

header_save_reason_byte_test() ->
    Cases = [{cold, 1}, {continued, 2}, {evict, 3}, {shutdown, 4}, {finish, 5}],
    [
        begin
            Meta = (base_meta())#{save_reason => Atom},
            {ok, Prefix} = erllama_cache_kvc:build(Meta, <<"x">>),
            <<_:5/binary, Byte:8, _/binary>> = Prefix,
            ?assertEqual({Atom, Expected}, {Atom, Byte})
        end
     || {Atom, Expected} <- Cases
    ].

header_payload_byte_count_test() ->
    Payload = binary:copy(<<"x">>, 12345),
    {ok, Prefix} = erllama_cache_kvc:build(base_meta(), Payload),
    <<_:40/binary, PayloadByteCount:64/little, _/binary>> = Prefix,
    ?assertEqual(12345, PayloadByteCount).

%% =============================================================================
%% Validation failures
%% =============================================================================

parse_rejects_bad_magic_test() ->
    {ok, Prefix} = erllama_cache_kvc:build(base_meta(), <<"x">>),
    Bad = <<"BAD", (binary:part(Prefix, 3, byte_size(Prefix) - 3))/binary, "x">>,
    Key = key_for(base_meta()),
    ?assertMatch({error, _}, erllama_cache_kvc:parse(Bad, Key)).

parse_rejects_bad_version_test() ->
    Meta = base_meta(),
    Payload = <<"x">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    %% Flip the version byte (offset 3) to 99.
    <<First:3/binary, _:8, Rest/binary>> = Prefix,
    Bad = <<First/binary, 99:8, Rest/binary, Payload/binary>>,
    Key = key_for(Meta),
    ?assertMatch(
        {error, {unsupported_version, 99}},
        erllama_cache_kvc:parse(Bad, Key)
    ).

parse_rejects_crc_mismatch_test() ->
    Meta = base_meta(),
    Payload = <<"original payload">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, "tampered payload">>,
    Key = key_for(Meta),
    ?assertMatch(
        {error, {crc_mismatch, _, _}},
        erllama_cache_kvc:parse(Bin, Key)
    ).

parse_rejects_key_mismatch_test() ->
    Meta = base_meta(),
    Payload = <<"x">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary>>,
    WrongKey = binary:copy(<<0>>, 32),
    ?assertMatch(
        {error, {key_mismatch, _, _}},
        erllama_cache_kvc:parse(Bin, WrongKey)
    ).

parse_rejects_truncated_test() ->
    {ok, Prefix} = erllama_cache_kvc:build(base_meta(), <<"x">>),
    Truncated = binary:part(Prefix, 0, 40),
    ?assertMatch({error, _}, erllama_cache_kvc:parse_meta(Truncated)).

parse_rejects_trailing_bytes_test() ->
    Meta = base_meta(),
    Payload = <<"x">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    Bin = <<Prefix/binary, Payload/binary, "extra">>,
    ?assertMatch({error, trailing_bytes}, erllama_cache_kvc:parse_meta(Bin)).

%% =============================================================================
%% parse_meta vs parse — meta path skips CRC/key
%% =============================================================================

parse_meta_does_not_check_crc_test() ->
    Meta = base_meta(),
    Payload = <<"original">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    %% Same length so the size invariants hold; only bytes change.
    Tampered = <<"tampered">>,
    Bin = <<Prefix/binary, Tampered/binary>>,
    {ok, _Info} = erllama_cache_kvc:parse_meta(Bin).

%% =============================================================================
%% Forward compatibility: unknown TLV tags
%% =============================================================================

parse_ignores_unknown_tlv_tags_test() ->
    Meta = base_meta(),
    Payload = <<"x">>,
    {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
    %% Inject a fake TLV entry by rewriting the TLV section.
    %% Layout: header(48) + trailer(24) + prompt_section + tlv_section.
    %% Locate TLV section by offset.
    <<Head:72/binary, PromptLen:32/little, _/binary>> = Prefix,
    PromptStart = 76,
    PromptText = binary:part(Prefix, PromptStart, PromptLen),
    TlvStart = PromptStart + PromptLen,
    <<_:TlvStart/binary, TlvLen:32/little, TlvBody:TlvLen/binary>> = Prefix,
    %% Append a fake unknown tag with arbitrary payload.
    UnknownEntry = <<16#FF:8, 4:32/little, "junk">>,
    NewTlvBody = <<TlvBody/binary, UnknownEntry/binary>>,
    NewTlvLen = byte_size(NewTlvBody),
    NewPrefix =
        <<Head/binary, PromptLen:32/little, PromptText/binary, NewTlvLen:32/little,
            NewTlvBody/binary>>,
    %% Trailer's payload_offset must be updated for the new TLV size.
    %% Recompute and rewrite the trailer.
    NewPayloadOffset = byte_size(NewPrefix),
    <<HBefore:48/binary, _OldPO:64/little, _OldPL:64/little, OldCrc:32/little, _Resv:32,
        AfterTrailer/binary>> = NewPrefix,
    PayloadLen = byte_size(Payload),
    NewTrailer = <<NewPayloadOffset:64/little, PayloadLen:64/little, OldCrc:32/little, 0:32>>,
    PrefixWithFixedTrailer = <<HBefore/binary, NewTrailer/binary, AfterTrailer/binary>>,
    Bin = <<PrefixWithFixedTrailer/binary, Payload/binary>>,
    Key = key_for(Meta),
    {ok, _Info, ParsedPayload} = erllama_cache_kvc:parse(Bin, Key),
    ?assertEqual(Payload, ParsedPayload).
