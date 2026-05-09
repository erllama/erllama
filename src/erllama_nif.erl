%% @doc
%% Single NIF entry module for erllama.
%%
%% v0.2 surface (post step 2b):
%%
%%   crc32c/1            CRC32C of an iodata, dirty CPU.
%%   fsync_dir/1         dir fsync (dirty IO).
%%   load_model/2        path + opts -> {ok, ModelRef} | {error, _}.
%%   free_model/1        eager release; resource also freed on GC.
%%   new_context/2       model + opts -> {ok, CtxRef} | {error, _}.
%%   free_context/1      eager release; resource also freed on GC.
%%   tokenize/3          model + text + opts -> [token_id()].
%%   kv_pack/3           ctx + tokens + n_tokens -> binary().
%%                       (Tokens/NTokens are informational; the in-
%%                       memory llama API saves whatever is currently
%%                       in the context's seq_id=0 KV cache. The model
%%                       layer prefills exactly the desired prefix
%%                       before calling.)
%%   kv_unpack/3         ctx + binary + seq_id -> ok | {error, _}.
%% @end
-module(erllama_nif).

-export([
    crc32c/1,
    fsync_dir/1,
    load_model/2,
    free_model/1,
    new_context/2,
    free_context/1,
    tokenize/3,
    detokenize/2,
    prefill/2,
    decode_one/1,
    kv_pack/3,
    kv_unpack/3,
    kv_seq_rm/4
]).

-on_load(init/0).

-export_type([model_ref/0, context_ref/0, token_id/0]).

-type model_ref() :: reference().
-type context_ref() :: reference().
-type token_id() :: integer().

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

%% =============================================================================
%% Public API
%% =============================================================================

-spec crc32c(iodata()) -> non_neg_integer().
crc32c(Data) -> nif_crc32c(Data).

-spec fsync_dir(iodata()) -> ok | {error, atom()}.
fsync_dir(Path) -> nif_fsync_dir(Path).

-spec load_model(iodata(), map()) -> {ok, model_ref()} | {error, atom()}.
load_model(Path, Opts) when is_map(Opts) -> nif_load_model(Path, Opts).

-spec free_model(model_ref()) -> ok.
free_model(Model) -> nif_free_model(Model).

-spec new_context(model_ref(), map()) -> {ok, context_ref()} | {error, atom()}.
new_context(Model, Opts) when is_map(Opts) -> nif_new_context(Model, Opts).

-spec free_context(context_ref()) -> ok.
free_context(Ctx) -> nif_free_context(Ctx).

-spec tokenize(model_ref(), iodata(), map()) -> [token_id()] | {error, atom()}.
tokenize(Model, Text, Opts) when is_map(Opts) -> nif_tokenize(Model, Text, Opts).

-spec detokenize(model_ref(), [token_id()]) -> binary() | {error, atom()}.
detokenize(Model, Tokens) -> nif_detokenize(Model, Tokens).

-spec prefill(context_ref(), [token_id()]) -> ok | {error, term()}.
prefill(Ctx, Tokens) -> nif_prefill(Ctx, Tokens).

-spec decode_one(context_ref()) ->
    {ok, token_id()} | {eog, token_id()} | {error, term()}.
decode_one(Ctx) -> nif_decode_one(Ctx).

-spec kv_pack(context_ref(), [token_id()], non_neg_integer()) ->
    binary() | {error, atom()}.
kv_pack(Ctx, Tokens, NTokens) -> nif_kv_pack(Ctx, Tokens, NTokens).

-spec kv_unpack(context_ref(), binary(), non_neg_integer()) ->
    ok | {error, atom()}.
kv_unpack(Ctx, Bin, SeqId) -> nif_kv_unpack(Ctx, Bin, SeqId).

%% Remove KV cells in [P0, P1) from sequence SeqId. Use P1 = -1 for
%% "to infinity". Required after kv_unpack to drop the last cell so
%% the corresponding token can be re-prefilled to regenerate logits.
-spec kv_seq_rm(context_ref(), integer(), integer(), integer()) ->
    ok | {error, atom()}.
kv_seq_rm(Ctx, SeqId, P0, P1) -> nif_kv_seq_rm(Ctx, SeqId, P0, P1).

%% =============================================================================
%% NIF stubs (replaced at on_load time)
%% =============================================================================

nif_crc32c(_Data) -> erlang:nif_error(nif_not_loaded).
nif_fsync_dir(_Path) -> erlang:nif_error(nif_not_loaded).
nif_load_model(_Path, _Opts) -> erlang:nif_error(nif_not_loaded).
nif_free_model(_Model) -> erlang:nif_error(nif_not_loaded).
nif_new_context(_Model, _Opts) -> erlang:nif_error(nif_not_loaded).
nif_free_context(_Ctx) -> erlang:nif_error(nif_not_loaded).
nif_tokenize(_Model, _Text, _Opts) -> erlang:nif_error(nif_not_loaded).
nif_detokenize(_Model, _Tokens) -> erlang:nif_error(nif_not_loaded).
nif_prefill(_Ctx, _Tokens) -> erlang:nif_error(nif_not_loaded).
nif_decode_one(_Ctx) -> erlang:nif_error(nif_not_loaded).
nif_kv_pack(_Ctx, _Tokens, _NTokens) -> erlang:nif_error(nif_not_loaded).
nif_kv_unpack(_Ctx, _Bin, _SeqId) -> erlang:nif_error(nif_not_loaded).
nif_kv_seq_rm(_Ctx, _SeqId, _P0, _P1) -> erlang:nif_error(nif_not_loaded).
