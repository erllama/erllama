%% @doc
%% Cache-key derivation.
%%
%% A cache key is the SHA-256 of model_fingerprint || quant_byte ||
%% ctx_params_hash || tokens_le32. Cache hits are token-exact by
%% construction; semantic / approximate matching is not allowed at
%% this layer.
%%
%% The quant byte is a stable cache-internal enumeration of
%% quantisation types. It is intentionally decoupled from llama.cpp's
%% GGUF tensor type IDs so we can add new entries without depending
%% on upstream renumbering.
%% @end
-module(erllama_cache_key).

-export([
    make/1,
    make/4,
    quant_byte/1,
    quant_atom/1,
    encode_tokens/1,
    decode_tokens/1
]).

-export_type([key/0, quant_type/0, components/0]).

-type key() :: <<_:256>>.
-type quant_type() ::
    f32
    | f16
    | q4_0
    | q4_1
    | q5_0
    | q5_1
    | q8_0
    | q4_k_m
    | q4_k_s
    | q5_k_m
    | q5_k_s
    | q6_k
    | q8_k.

-type components() :: #{
    fingerprint := <<_:256>>,
    quant_type := quant_type(),
    ctx_params_hash := <<_:256>>,
    tokens := [non_neg_integer()]
}.

-spec make(components()) -> key().
make(#{
    fingerprint := Fp,
    quant_type := QT,
    ctx_params_hash := CtxHash,
    tokens := Tokens
}) when
    is_binary(Fp),
    byte_size(Fp) =:= 32,
    is_binary(CtxHash),
    byte_size(CtxHash) =:= 32,
    is_list(Tokens)
->
    make(Fp, QT, CtxHash, encode_tokens(Tokens)).

%% @doc Variant taking a pre-encoded TokensBin (u32-LE per token,
%% matching `encode_tokens/1`). Used by the longest-prefix walk so a
%% caller can encode once and pass `binary:part(AllTokensBin, 0, N*4)`
%% sub-binaries per probe, avoiding the per-attempt list traversal +
%% list comprehension allocation. Sub-binaries are O(1) views, so
%% this turns the per-probe cost into just the SHA-256 work.
-spec make(<<_:256>>, quant_type(), <<_:256>>, binary()) -> key().
make(Fp, QT, CtxHash, TokensBin) when
    is_binary(Fp),
    byte_size(Fp) =:= 32,
    is_binary(CtxHash),
    byte_size(CtxHash) =:= 32,
    is_binary(TokensBin),
    byte_size(TokensBin) rem 4 =:= 0
->
    QuantByte = quant_byte(QT),
    crypto:hash(sha256, [Fp, <<QuantByte:8>>, CtxHash, TokensBin]).

-spec quant_byte(quant_type()) -> 0..255.
quant_byte(f32) -> 0;
quant_byte(f16) -> 1;
quant_byte(q4_0) -> 2;
quant_byte(q4_1) -> 3;
quant_byte(q5_0) -> 4;
quant_byte(q5_1) -> 5;
quant_byte(q8_0) -> 6;
quant_byte(q4_k_m) -> 7;
quant_byte(q4_k_s) -> 8;
quant_byte(q5_k_m) -> 9;
quant_byte(q5_k_s) -> 10;
quant_byte(q6_k) -> 11;
quant_byte(q8_k) -> 12.

-spec quant_atom(0..255) -> {ok, quant_type()} | {error, unknown_quant}.
quant_atom(0) -> {ok, f32};
quant_atom(1) -> {ok, f16};
quant_atom(2) -> {ok, q4_0};
quant_atom(3) -> {ok, q4_1};
quant_atom(4) -> {ok, q5_0};
quant_atom(5) -> {ok, q5_1};
quant_atom(6) -> {ok, q8_0};
quant_atom(7) -> {ok, q4_k_m};
quant_atom(8) -> {ok, q4_k_s};
quant_atom(9) -> {ok, q5_k_m};
quant_atom(10) -> {ok, q5_k_s};
quant_atom(11) -> {ok, q6_k};
quant_atom(12) -> {ok, q8_k};
quant_atom(_) -> {error, unknown_quant}.

-spec encode_tokens([non_neg_integer()]) -> binary().
encode_tokens(Tokens) ->
    <<<<T:32/little>> || T <- Tokens>>.

-spec decode_tokens(binary()) -> [non_neg_integer()].
decode_tokens(Bin) when is_binary(Bin), byte_size(Bin) rem 4 =:= 0 ->
    [T || <<T:32/little>> <= Bin].
