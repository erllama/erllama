%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(prop_erllama_cache_kvc).
-include_lib("proper/include/proper.hrl").

quant_type() ->
    elements([
        f32,
        f16,
        q4_0,
        q4_1,
        q5_0,
        q5_1,
        q8_0,
        q4_k_m,
        q4_k_s,
        q5_k_m,
        q5_k_s,
        q6_k,
        q8_k
    ]).

fingerprint_mode() ->
    elements([safe, gguf_chunked, fast_unsafe]).

save_reason() ->
    elements([cold, continued, finish, evict, shutdown]).

quant_bits() ->
    elements([2, 4, 5, 6, 8, 16, 32]).

bin32() ->
    ?LET(B, vector(32, range(0, 255)), list_to_binary(B)).

short_bin() ->
    ?LET(B, list(range(0, 255)), list_to_binary(B)).

token_id() ->
    range(0, 16#FFFFFFFF).

valid_meta() ->
    ?LET(
        {SR, QB, Fp, FpMode, QT, CtxHash, Tokens, CtxSize, Prompt, Host, Vsn},
        {
            save_reason(),
            quant_bits(),
            bin32(),
            fingerprint_mode(),
            quant_type(),
            bin32(),
            list(token_id()),
            pos_integer(),
            short_bin(),
            short_bin(),
            short_bin()
        },
        #{
            save_reason => SR,
            quant_bits => QB,
            fingerprint => Fp,
            fingerprint_mode => FpMode,
            quant_type => QT,
            ctx_params_hash => CtxHash,
            tokens => Tokens,
            context_size => CtxSize,
            creation_time => 1000,
            last_used_time => 2000,
            hit_count => 0,
            prompt_text => Prompt,
            hostname => Host,
            erllama_version => Vsn
        }
    ).

key_for(Meta) ->
    erllama_cache_key:make(#{
        fingerprint => maps:get(fingerprint, Meta),
        quant_type => maps:get(quant_type, Meta),
        ctx_params_hash => maps:get(ctx_params_hash, Meta),
        tokens => maps:get(tokens, Meta)
    }).

%% Round-trip: build then parse returns the same payload and the
%% cache_key we expect.
prop_round_trip() ->
    ?FORALL(
        {Meta, Payload},
        {valid_meta(), short_bin()},
        begin
            {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
            Bin = <<Prefix/binary, Payload/binary>>,
            Key = key_for(Meta),
            case erllama_cache_kvc:parse(Bin, Key) of
                {ok, _Info, P} -> P =:= Payload;
                _ -> false
            end
        end
    ).

%% Header always starts with the magic bytes and version 1.
prop_header_magic_and_version() ->
    ?FORALL(
        {Meta, Payload},
        {valid_meta(), short_bin()},
        begin
            {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
            <<Magic:3/binary, Version:8, _/binary>> = Prefix,
            Magic =:= <<"KVC">> andalso Version =:= 1
        end
    ).

%% Parsed info preserves the token list verbatim, in order.
prop_tokens_round_trip() ->
    ?FORALL(
        {Meta, Payload},
        {valid_meta(), short_bin()},
        begin
            {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
            Bin = <<Prefix/binary, Payload/binary>>,
            {ok, Info} = erllama_cache_kvc:parse_meta(Bin),
            maps:get(tokens, Info) =:= maps:get(tokens, Meta)
        end
    ).

%% A wrong expected key always rejects, even if everything else is fine.
prop_wrong_key_rejects() ->
    ?FORALL(
        {Meta, Payload, WrongByte},
        {valid_meta(), short_bin(), range(1, 255)},
        begin
            {ok, Prefix} = erllama_cache_kvc:build(Meta, Payload),
            Bin = <<Prefix/binary, Payload/binary>>,
            CorrectKey = key_for(Meta),
            WrongKey = binary:copy(<<WrongByte>>, 32),
            CorrectKey =:= WrongKey orelse
                element(1, erllama_cache_kvc:parse(Bin, WrongKey)) =:= error
        end
    ).
