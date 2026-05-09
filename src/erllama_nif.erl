%% @doc
%% Single NIF entry module for erllama.
%%
%% Surface in v0.1:
%%
%%   crc32c/1     CRC32C of an iodata, computed on a dirty CPU
%%                scheduler. The cache layer calls this once per save
%%                (before publication) and once per load (during
%%                validation).
%%
%%   kv_pack/3    Stub returning {error, not_implemented} until the
%%   kv_unpack/3  llama.cpp wiring lands. Both will read/write
%%                `llama_context*` state via llama.cpp's
%%                `llama_state_seq_*` family.
%%
%% Future additions in this module (one .so for the whole project):
%% load_model/3, free_model/1, new_context/2, free_context/1,
%% tokenize/2, detokenize/2, prefill/2, decode_async/3.
%%
%% No file I/O happens in any of these NIFs. The cache layer assembles
%% framed `.kvc` files in pure Erlang and only feeds opaque payload
%% bytes to the NIF.
%% @end
-module(erllama_nif).

-export([crc32c/1, kv_pack/3, kv_unpack/3]).

-on_load(init/0).

-spec init() -> ok | {error, term()}.
init() ->
    PrivDir =
        case code:priv_dir(erllama) of
            {error, bad_name} ->
                EbinDir = filename:dirname(code:which(?MODULE)),
                filename:join(filename:dirname(EbinDir), "priv");
            Dir ->
                Dir
        end,
    SoName = filename:join(PrivDir, "erllama_nif"),
    erlang:load_nif(SoName, 0).

-spec crc32c(iodata()) -> non_neg_integer().
crc32c(Data) -> nif_crc32c(Data).

-spec kv_pack(reference(), [non_neg_integer()], non_neg_integer()) ->
    binary() | {error, not_implemented}.
kv_pack(Ctx, Tokens, NTokens) -> nif_kv_pack(Ctx, Tokens, NTokens).

-spec kv_unpack(reference(), binary(), non_neg_integer()) ->
    ok | {error, term()}.
kv_unpack(Ctx, Bin, SeqId) -> nif_kv_unpack(Ctx, Bin, SeqId).

%% =============================================================================
%% NIF stubs (replaced at on_load time)
%% =============================================================================

nif_crc32c(_Data) ->
    erlang:nif_error(nif_not_loaded).

nif_kv_pack(_Ctx, _Tokens, _NTokens) ->
    erlang:nif_error(nif_not_loaded).

nif_kv_unpack(_Ctx, _Bin, _SeqId) ->
    erlang:nif_error(nif_not_loaded).
