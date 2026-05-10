%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_kvc).
-moduledoc """
KVC v2 file framing and TLV codec.

File layout (little-endian throughout):

```
[0..47]    Header (48 bytes, ds4-compatible):
              u8[3]  magic = "KVC"
              u8     version = 1
              u8     quant_bits
              u8     save_reason (0..5)
              u8[2]  reserved
              u32    cached_token_count
              u32    hit_count
              u32    context_size
              u8[4]  reserved
              u64    creation_time (unix seconds)
              u64    last_used_time (unix seconds)
              u64    payload_byte_count

[48..71]   Trailer (24 bytes, erllama-specific):
              u64    payload_offset
              u64    payload_length
              u32    payload_crc32c
              u32    reserved

[72..]     Prompt section: u32_le length, then UTF-8 bytes
           (observability only; untrusted on load)

[..]       TLV section: u32_le length, then a sequence of
           (u8 tag, u32_le length, value) records

[..end]    Payload: opaque bytes (raw llama_state_seq blob)
```

TLV tags:

```
0x01 fingerprint        32 bytes
0x02 fingerprint_mode   1 byte (0=safe, 1=gguf_chunked, 2=fast_unsafe)
0x03 quant_type         1 byte (cache-internal byte; see erllama_cache_key)
0x04 ctx_params_hash    32 bytes
0x05 hostname           variable
0x06 erllama_version    variable
0x07 save_reason_detail variable
0x08 token_id_count     u32_le
0x09 token_ids          variable (u32_le * count)
```

Build path is split: callers pass the payload binary separately so
it never gets concatenated into a larger BEAM allocation. The
returned prefix is the bytes preceding the payload; combined with
the payload they form a complete file.

Parse paths come in two flavours:
  parse_meta/1 - header + trailer + TLV only, no payload CRC.
                 Used by tier on-start scans which defer CRC
                 verification until the file is actually loaded.
  parse/2      - full validation including payload CRC32C and
                 cache_key replay against an expected key.
""".

-include("erllama_cache.hrl").

-export([build/2, parse/2, parse_meta/1]).

-export_type([build_meta/0, info/0]).

%% =============================================================================
%% Types
%% =============================================================================

-type save_reason() :: unknown | cold | continued | evict | shutdown | finish.

-type fingerprint_mode() :: safe | gguf_chunked | fast_unsafe.

-type build_meta() :: #{
    save_reason := save_reason(),
    quant_bits := non_neg_integer(),
    fingerprint := <<_:256>>,
    fingerprint_mode := fingerprint_mode(),
    quant_type := erllama_cache_key:quant_type(),
    ctx_params_hash := <<_:256>>,
    tokens := [non_neg_integer()],
    context_size := non_neg_integer(),
    creation_time => non_neg_integer(),
    last_used_time => non_neg_integer(),
    hit_count => non_neg_integer(),
    prompt_text => binary(),
    hostname => binary(),
    erllama_version => binary(),
    save_reason_detail => binary()
}.

-type info() :: #{
    magic := binary(),
    version := non_neg_integer(),
    save_reason := save_reason(),
    quant_bits := non_neg_integer(),
    cached_token_count := non_neg_integer(),
    hit_count := non_neg_integer(),
    context_size := non_neg_integer(),
    creation_time := non_neg_integer(),
    last_used_time := non_neg_integer(),
    payload_byte_count := non_neg_integer(),
    payload_offset := non_neg_integer(),
    payload_length := non_neg_integer(),
    payload_crc32c := non_neg_integer(),
    prompt_text := binary(),
    fingerprint := binary(),
    fingerprint_mode := fingerprint_mode(),
    quant_type := erllama_cache_key:quant_type(),
    ctx_params_hash := binary(),
    tokens := [non_neg_integer()],
    hostname => binary(),
    erllama_version => binary(),
    save_reason_detail => binary()
}.

%% =============================================================================
%% Format constants
%% =============================================================================

-define(MAGIC, <<"KVC">>).
-define(VERSION, 1).
-define(HEADER_SIZE, 48).
-define(TRAILER_SIZE, 24).

-define(SR_UNKNOWN, 0).
-define(SR_COLD, 1).
-define(SR_CONTINUED, 2).
-define(SR_EVICT, 3).
-define(SR_SHUTDOWN, 4).
-define(SR_FINISH, 5).

-define(FM_SAFE, 0).
-define(FM_CHUNKED, 1).
-define(FM_FAST_UNSAFE, 2).

-define(TLV_FINGERPRINT, 16#01).
-define(TLV_FINGERPRINT_MODE, 16#02).
-define(TLV_QUANT_TYPE, 16#03).
-define(TLV_CTX_PARAMS_HASH, 16#04).
-define(TLV_HOSTNAME, 16#05).
-define(TLV_ERLLAMA_VERSION, 16#06).
-define(TLV_SAVE_REASON_DETAIL, 16#07).
-define(TLV_TOKEN_ID_COUNT, 16#08).
-define(TLV_TOKEN_IDS, 16#09).

%% =============================================================================
%% Build
%% =============================================================================

-spec build(build_meta(), binary()) -> {ok, binary()} | {error, term()}.
build(Meta, Payload) when is_map(Meta), is_binary(Payload) ->
    try
        do_build(Meta, Payload)
    catch
        error:{badmap, _} = E -> {error, E};
        error:{badkey, K} -> {error, {missing_key, K}};
        error:Reason -> {error, Reason}
    end.

do_build(Meta, Payload) ->
    Tokens = maps:get(tokens, Meta),
    PayloadCrc = erllama_nif:crc32c(Payload),
    PayloadLen = byte_size(Payload),

    PromptText = maps:get(prompt_text, Meta, <<>>),
    PromptSection = <<(byte_size(PromptText)):32/little, PromptText/binary>>,

    Tlv = build_tlv(Meta, Tokens),
    TlvSection = <<(byte_size(Tlv)):32/little, Tlv/binary>>,

    PayloadOffset =
        ?HEADER_SIZE + ?TRAILER_SIZE +
            byte_size(PromptSection) +
            byte_size(TlvSection),

    Header = build_header(Meta, length(Tokens), PayloadLen),
    Trailer = build_trailer(PayloadOffset, PayloadLen, PayloadCrc),

    Prefix =
        <<Header/binary, Trailer/binary, PromptSection/binary, TlvSection/binary>>,
    {ok, Prefix}.

build_header(Meta, TokenCount, PayloadLen) ->
    QuantBits = maps:get(quant_bits, Meta),
    SaveReasonByte = save_reason_byte(maps:get(save_reason, Meta)),
    HitCount = maps:get(hit_count, Meta, 0),
    ContextSize = maps:get(context_size, Meta),
    Now = erlang:system_time(second),
    CreationTime = maps:get(creation_time, Meta, Now),
    LastUsedTime = maps:get(last_used_time, Meta, Now),
    <<?MAGIC/binary, ?VERSION:8, QuantBits:8, SaveReasonByte:8, 0:16, TokenCount:32/little,
        HitCount:32/little, ContextSize:32/little, 0:32, CreationTime:64/little,
        LastUsedTime:64/little, PayloadLen:64/little>>.

build_trailer(PayloadOffset, PayloadLen, PayloadCrc) ->
    <<PayloadOffset:64/little, PayloadLen:64/little, PayloadCrc:32/little, 0:32>>.

build_tlv(Meta, Tokens) ->
    QuantTypeByte = erllama_cache_key:quant_byte(maps:get(quant_type, Meta)),
    FpModeByte = fingerprint_mode_byte(maps:get(fingerprint_mode, Meta)),
    Required = [
        {?TLV_FINGERPRINT, maps:get(fingerprint, Meta)},
        {?TLV_FINGERPRINT_MODE, <<FpModeByte:8>>},
        {?TLV_QUANT_TYPE, <<QuantTypeByte:8>>},
        {?TLV_CTX_PARAMS_HASH, maps:get(ctx_params_hash, Meta)},
        {?TLV_TOKEN_ID_COUNT, <<(length(Tokens)):32/little>>},
        {?TLV_TOKEN_IDS, erllama_cache_key:encode_tokens(Tokens)}
    ],
    Optional = [
        {?TLV_HOSTNAME, default_hostname(Meta)},
        {?TLV_ERLLAMA_VERSION, default_erllama_version(Meta)},
        {?TLV_SAVE_REASON_DETAIL, maps:get(save_reason_detail, Meta, undefined)}
    ],
    All = Required ++ [E || {_, V} = E <- Optional, V =/= undefined, V =/= <<>>],
    iolist_to_binary([
        <<Tag:8, (byte_size(V)):32/little, V/binary>>
     || {Tag, V} <- All
    ]).

default_hostname(Meta) ->
    case maps:get(hostname, Meta, undefined) of
        undefined ->
            {ok, H} = inet:gethostname(),
            list_to_binary(H);
        H ->
            H
    end.

default_erllama_version(Meta) ->
    case maps:get(erllama_version, Meta, undefined) of
        undefined ->
            {ok, V} = application:get_key(erllama, vsn),
            list_to_binary(V);
        V ->
            V
    end.

%% =============================================================================
%% Parse (meta only, no payload CRC)
%% =============================================================================

-spec parse_meta(binary()) -> {ok, info()} | {error, term()}.
parse_meta(Bin) when is_binary(Bin) ->
    try
        {Header, Rest1} = take_header(Bin),
        {Trailer, Rest2} = take_trailer(Rest1),
        {PromptText, Rest3} = take_length_prefixed(Rest2),
        {Tlv, _Rest4} = take_length_prefixed(Rest3),
        TlvFields = decode_tlv(Tlv),
        Info0 = maps:merge(Header, Trailer),
        Info1 = Info0#{prompt_text => PromptText},
        Info = maps:merge(Info1, TlvFields),
        validate_meta_consistency(Info, byte_size(Bin)),
        {ok, Info}
    catch
        throw:{kvc_error, Reason} ->
            {error, Reason};
        error:{badmatch, _} ->
            {error, malformed}
    end.

%% =============================================================================
%% Parse (full, including CRC + key replay)
%% =============================================================================

-spec parse(binary(), erllama_cache_key:key()) ->
    {ok, info(), binary()} | {error, term()}.
parse(Bin, ExpectedKey) when is_binary(Bin), is_binary(ExpectedKey) ->
    case parse_meta(Bin) of
        {error, _} = E ->
            E;
        {ok, Info} ->
            #{
                payload_offset := PO,
                payload_length := PL,
                payload_crc32c := ExpectedCrc
            } = Info,
            case Bin of
                <<_:PO/binary, Payload:PL/binary, _/binary>> ->
                    verify(Info, Payload, ExpectedCrc, ExpectedKey);
                _ ->
                    {error, payload_out_of_bounds}
            end
    end.

verify(Info, Payload, ExpectedCrc, ExpectedKey) ->
    ActualCrc = erllama_nif:crc32c(Payload),
    case ActualCrc =:= ExpectedCrc of
        false ->
            {error, {crc_mismatch, ExpectedCrc, ActualCrc}};
        true ->
            ActualKey = erllama_cache_key:make(#{
                fingerprint => maps:get(fingerprint, Info),
                quant_type => maps:get(quant_type, Info),
                ctx_params_hash => maps:get(ctx_params_hash, Info),
                tokens => maps:get(tokens, Info)
            }),
            case ActualKey =:= ExpectedKey of
                false -> {error, {key_mismatch, ExpectedKey, ActualKey}};
                true -> {ok, Info, Payload}
            end
    end.

%% =============================================================================
%% Internal: framing helpers
%% =============================================================================

take_header(<<
    "KVC",
    Version:8,
    QuantBits:8,
    SaveReasonByte:8,
    _:16,
    CachedTokenCount:32/little,
    HitCount:32/little,
    ContextSize:32/little,
    _:32,
    CreationTime:64/little,
    LastUsedTime:64/little,
    PayloadByteCount:64/little,
    Rest/binary
>>) ->
    case Version =:= ?VERSION of
        false ->
            throw({kvc_error, {unsupported_version, Version}});
        true ->
            ok
    end,
    Header = #{
        magic => ?MAGIC,
        version => Version,
        quant_bits => QuantBits,
        save_reason => save_reason_atom(SaveReasonByte),
        cached_token_count => CachedTokenCount,
        hit_count => HitCount,
        context_size => ContextSize,
        creation_time => CreationTime,
        last_used_time => LastUsedTime,
        payload_byte_count => PayloadByteCount
    },
    {Header, Rest};
take_header(_) ->
    throw({kvc_error, bad_header}).

take_trailer(<<
    PayloadOffset:64/little,
    PayloadLength:64/little,
    PayloadCrc32c:32/little,
    _:32,
    Rest/binary
>>) ->
    Trailer = #{
        payload_offset => PayloadOffset,
        payload_length => PayloadLength,
        payload_crc32c => PayloadCrc32c
    },
    {Trailer, Rest};
take_trailer(_) ->
    throw({kvc_error, bad_trailer}).

take_length_prefixed(<<Len:32/little, Bin:Len/binary, Rest/binary>>) ->
    {Bin, Rest};
take_length_prefixed(_) ->
    throw({kvc_error, bad_length_prefixed_section}).

decode_tlv(Bin) ->
    decode_tlv(Bin, #{}).

decode_tlv(<<>>, Acc) ->
    Acc;
decode_tlv(<<Tag:8, Len:32/little, Value:Len/binary, Rest/binary>>, Acc) ->
    decode_tlv(Rest, store_tlv(Tag, Value, Acc));
decode_tlv(_, _) ->
    throw({kvc_error, bad_tlv}).

store_tlv(?TLV_FINGERPRINT, V, Acc) when byte_size(V) =:= 32 ->
    Acc#{fingerprint => V};
store_tlv(?TLV_FINGERPRINT_MODE, <<B:8>>, Acc) ->
    Acc#{fingerprint_mode => fingerprint_mode_atom(B)};
store_tlv(?TLV_QUANT_TYPE, <<B:8>>, Acc) ->
    case erllama_cache_key:quant_atom(B) of
        {ok, Q} -> Acc#{quant_type => Q};
        {error, unknown_quant} -> throw({kvc_error, {unknown_quant_byte, B}})
    end;
store_tlv(?TLV_CTX_PARAMS_HASH, V, Acc) when byte_size(V) =:= 32 ->
    Acc#{ctx_params_hash => V};
store_tlv(?TLV_HOSTNAME, V, Acc) ->
    Acc#{hostname => V};
store_tlv(?TLV_ERLLAMA_VERSION, V, Acc) ->
    Acc#{erllama_version => V};
store_tlv(?TLV_SAVE_REASON_DETAIL, V, Acc) ->
    Acc#{save_reason_detail => V};
store_tlv(?TLV_TOKEN_ID_COUNT, <<_N:32/little>>, Acc) ->
    %% Validation only; the canonical count comes from token bytes.
    Acc;
store_tlv(?TLV_TOKEN_IDS, V, Acc) when byte_size(V) rem 4 =:= 0 ->
    Acc#{tokens => erllama_cache_key:decode_tokens(V)};
store_tlv(_Tag, _V, Acc) ->
    %% Forward-compatible: ignore unknown tags.
    Acc.

validate_meta_consistency(Info, TotalSize) ->
    #{
        payload_offset := PO,
        payload_length := PL,
        payload_byte_count := PBC
    } = Info,
    case PL =:= PBC of
        false -> throw({kvc_error, payload_length_mismatch});
        true -> ok
    end,
    case TotalSize >= PO + PL of
        false -> throw({kvc_error, truncated});
        true -> ok
    end,
    case TotalSize =:= PO + PL of
        true -> ok;
        false -> throw({kvc_error, trailing_bytes})
    end,
    require_field(fingerprint, Info),
    require_field(fingerprint_mode, Info),
    require_field(quant_type, Info),
    require_field(ctx_params_hash, Info),
    require_field(tokens, Info),
    ok.

require_field(K, Info) ->
    case maps:is_key(K, Info) of
        true -> ok;
        false -> throw({kvc_error, {missing_tlv, K}})
    end.

%% =============================================================================
%% Internal: enumerations
%% =============================================================================

save_reason_byte(unknown) -> ?SR_UNKNOWN;
save_reason_byte(cold) -> ?SR_COLD;
save_reason_byte(continued) -> ?SR_CONTINUED;
save_reason_byte(evict) -> ?SR_EVICT;
save_reason_byte(shutdown) -> ?SR_SHUTDOWN;
save_reason_byte(finish) -> ?SR_FINISH.

save_reason_atom(?SR_UNKNOWN) -> unknown;
save_reason_atom(?SR_COLD) -> cold;
save_reason_atom(?SR_CONTINUED) -> continued;
save_reason_atom(?SR_EVICT) -> evict;
save_reason_atom(?SR_SHUTDOWN) -> shutdown;
save_reason_atom(?SR_FINISH) -> finish;
save_reason_atom(B) -> throw({kvc_error, {unknown_save_reason, B}}).

fingerprint_mode_byte(safe) -> ?FM_SAFE;
fingerprint_mode_byte(gguf_chunked) -> ?FM_CHUNKED;
fingerprint_mode_byte(fast_unsafe) -> ?FM_FAST_UNSAFE.

fingerprint_mode_atom(?FM_SAFE) -> safe;
fingerprint_mode_atom(?FM_CHUNKED) -> gguf_chunked;
fingerprint_mode_atom(?FM_FAST_UNSAFE) -> fast_unsafe;
fingerprint_mode_atom(B) -> throw({kvc_error, {unknown_fingerprint_mode, B}}).
