%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
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
    kv_pack/4,
    kv_unpack/3,
    kv_seq_rm/4,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    adapter_load/2,
    adapter_free/1,
    set_adapters/2,
    sampler_new/2,
    sampler_free/1,
    vram_info/0,
    model_size/1,
    model_n_layer/1,
    forward_with_argmax/2
]).

-export_type([adapter_ref/0, sampler_ref/0]).

-on_load(init/0).

-export_type([model_ref/0, context_ref/0, token_id/0]).

-type model_ref() :: reference().
-type context_ref() :: reference().
-type adapter_ref() :: reference().
-type sampler_ref() :: reference().
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

-doc """
Load a GGUF model.

Recognised keys in `Opts` (all optional; defaults come from
`llama_model_default_params()`):

- `n_gpu_layers :: integer()` — number of layers offloaded to GPU.
- `main_gpu :: non_neg_integer()` — GPU index when `split_mode = none`.
- `split_mode :: none | layer | row` — how to split a model across
  multiple GPUs. Atom mapping: `none -> LLAMA_SPLIT_MODE_NONE`,
  `layer -> LLAMA_SPLIT_MODE_LAYER`, `row -> LLAMA_SPLIT_MODE_ROW`.
- `tensor_split :: [float()]` — per-device proportions when splitting.
  Up to `llama_max_devices()` entries (16 in the vendored llama.cpp);
  shorter lists zero-fill the tail.
- `use_mmap, use_mlock, vocab_only :: boolean()`.

A bad atom for `split_mode`, or a non-numeric entry in
`tensor_split`, raises `badarg`.
""".
-spec load_model(iodata(), map()) -> {ok, model_ref()} | {error, atom()}.
load_model(Path, Opts) when is_map(Opts) -> nif_load_model(Path, Opts).

-spec free_model(model_ref()) -> ok.
free_model(Model) -> nif_free_model(Model).

-doc """
Build a new inference context against a loaded model.

Recognised keys in `Opts` (all optional; defaults come from
`llama_context_default_params()`):

- `n_ctx, n_batch, n_ubatch, n_seq_max :: pos_integer()`.
- `n_threads, n_threads_batch :: pos_integer()`.
- `embeddings, offload_kqv :: boolean()`.
- `flash_attn :: boolean() | auto` — `true` enables, `false`
  disables, `auto` lets llama.cpp decide based on the build and
  model. Maps to `enum llama_flash_attn_type`.
- `type_k, type_v :: f16 | f32 | bf16 | q4_0 | q5_0 | q5_1 | q8_0`
  — KV cache element type for keys and values. Maps to
  `GGML_TYPE_*`.

A bad atom for any of `flash_attn`, `type_k`, or `type_v` raises
`badarg`.
""".
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

%% Seq-aware kv_pack. Extract the KV state for a specific seq_id.
%% Used by multi-sequence batching (v0.2+); existing v0.1 callers
%% stay on the 3-arity which defaults to seq_id=0.
-spec kv_pack(context_ref(), [token_id()], non_neg_integer(), non_neg_integer()) ->
    binary() | {error, atom()}.
kv_pack(Ctx, Tokens, NTokens, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    nif_kv_pack(Ctx, Tokens, NTokens, SeqId).

-spec kv_unpack(context_ref(), binary(), non_neg_integer()) ->
    ok | {error, atom()}.
kv_unpack(Ctx, Bin, SeqId) -> nif_kv_unpack(Ctx, Bin, SeqId).

%% Remove KV cells in [P0, P1) from sequence SeqId. Use P1 = -1 for
%% "to infinity". Required after kv_unpack to drop the last cell so
%% the corresponding token can be re-prefilled to regenerate logits.
-spec kv_seq_rm(context_ref(), integer(), integer(), integer()) ->
    ok | {error, atom()}.
kv_seq_rm(Ctx, SeqId, P0, P1) -> nif_kv_seq_rm(Ctx, SeqId, P0, P1).

%% Render a normalised chat request through the model's chat template
%% (read from GGUF metadata) and tokenise the result. Request is a map
%% with `messages`, optional `system`, optional `tools`. Returns a
%% list of token ids on success.
-spec apply_chat_template(model_ref(), map()) ->
    {ok, [token_id()]} | {error, atom()}.
apply_chat_template(Model, Request) when is_map(Request) ->
    nif_apply_chat_template(Model, Request).

%% Decode a token list and read the per-sequence pooled embedding
%% vector. The context must have been opened with `embeddings => true`.
-spec embed(context_ref(), [token_id()]) ->
    {ok, [float()]} | {error, atom()}.
embed(Ctx, Tokens) when is_list(Tokens) ->
    nif_embed(Ctx, Tokens).

%% Install a GBNF grammar on the context's sampler. Subsequent
%% `decode_one/1` calls sample only tokens that keep the output on a
%% valid grammar path. Use `clear_sampler/1` to drop the grammar
%% (returns the context to greedy sampling on the next decode).
%%
%% Equivalent to `configure_sampler(Ctx, #{grammar => Grammar})`.
-spec set_grammar(context_ref(), binary()) -> ok | {error, atom()}.
set_grammar(Ctx, Grammar) when is_binary(Grammar) ->
    nif_set_grammar(Ctx, Grammar).

%% Build the sampler chain in one shot from a config map. Recognised
%% keys (all optional):
%%
%%   grammar             :: binary()           %% GBNF source
%%   repetition_penalty  :: float()            %% > 1.0 penalises repeats
%%   top_k               :: non_neg_integer()
%%   top_p               :: float()            %% (0, 1]
%%   min_p               :: float()            %% (0, 1]
%%   temperature         :: float()            %% 0.0 == greedy
%%   seed                :: non_neg_integer()  %% honoured only with temperature > 0
%%
%% Stages are appended in a deterministic order:
%% grammar -> repetition_penalty -> top_k -> top_p -> min_p ->
%% (temperature > 0 ? temp -> dist(seed) : greedy).
%%
%% Replaces any previously configured chain on the context atomically.
-spec configure_sampler(context_ref(), map()) -> ok | {error, atom()}.
configure_sampler(Ctx, Cfg) when is_map(Cfg) ->
    nif_configure_sampler(Ctx, Cfg).

-spec clear_sampler(context_ref()) -> ok.
clear_sampler(Ctx) ->
    nif_clear_sampler(Ctx).

%% Load a LoRA adapter from a GGUF file. Bound to the model: the
%% adapter is freed when the model is, or earlier on
%% `adapter_free/1`. The model is keep-referenced by the adapter
%% resource so `free_model/1` returns `{ok, deferred}` until all
%% attached adapters are dropped.
-spec adapter_load(model_ref(), iodata()) ->
    {ok, adapter_ref()} | {error, atom()}.
adapter_load(Model, Path) ->
    nif_adapter_load(Model, Path).

%% Explicit free. Idempotent: a second call returns
%% `{error, released}`. The implicit destructor handles the case
%% where the user drops the reference without calling free.
-spec adapter_free(adapter_ref()) -> ok | {error, atom()}.
adapter_free(Adapter) ->
    nif_adapter_free(Adapter).

%% Install a list of {adapter_ref(), Scale} pairs on the context.
%% Replaces any previously installed set; passing [] detaches
%% everything.
-spec set_adapters(context_ref(), [{adapter_ref(), float()}]) ->
    ok | {error, atom()}.
set_adapters(Ctx, Adapters) when is_list(Adapters) ->
    nif_set_adapters(Ctx, Adapters).

%% Build a standalone sampler chain from the same config map
%% configure_sampler/2 accepts. Holds a keep-reference on the
%% context so the context stays alive at least as long as the
%% sampler. v0.1 callers don't need this - it's the building block
%% for multi-seq batching coming in v0.2 (one sampler per request).
-spec sampler_new(context_ref(), map()) ->
    {ok, sampler_ref()} | {error, atom()}.
sampler_new(Ctx, Cfg) when is_map(Cfg) ->
    nif_sampler_new(Ctx, Cfg).

%% Explicit free. Idempotent: a second call returns
%% `{error, released}`. The implicit destructor handles unfreed
%% samplers when the resource is garbage-collected.
-spec sampler_free(sampler_ref()) -> ok | {error, atom()}.
sampler_free(Sampler) ->
    nif_sampler_free(Sampler).

%% Walk every loaded ggml backend and sum free / total memory across
%% non-CPU devices (GPU, integrated GPU, accelerator). Returns
%% `{error, no_gpu}` on a CPU-only build (no faked numbers). Used by
%% the cluster scheduler for bin-packing model placement.
-spec vram_info() ->
    {ok, #{
        total_b := non_neg_integer(),
        free_b := non_neg_integer(),
        used_b := non_neg_integer()
    }}
    | {error, atom()}.
vram_info() ->
    nif_vram_info().

%% Total byte size of the loaded model on disk. Used to derive
%% vram_estimate_b for list_models metadata. 0 on exception.
-spec model_size(model_ref()) -> non_neg_integer() | {error, atom()}.
model_size(Model) ->
    nif_model_size(Model).

%% Total layer count of the loaded model. Used together with
%% n_gpu_layers to compute the offload fraction for vram_estimate_b.
-spec model_n_layer(model_ref()) -> non_neg_integer() | {error, atom()}.
model_n_layer(Model) ->
    nif_model_n_layer(Model).

%% Per-position argmax over the model vocab. Decodes `Tokens` with
%% logits flagged on every position, then returns the argmax id at
%% each position (mapped to the atom `eos` for end-of-generation
%% tokens). Used by erllama:verify/4 for speculative-decoding
%% candidate verification.
%%
%% Mutates KV state: after return, the context's seq_id=0 KV cells
%% extend by length(Tokens). Callers that need to roll back must
%% snapshot (KV length, decode_ready, last token) before the call
%% and restore via kv_seq_rm + a re-prefill of the last pre-call
%% token.
-spec forward_with_argmax(context_ref(), [token_id()]) ->
    {ok, [token_id() | eos]} | {error, atom()}.
forward_with_argmax(Ctx, Tokens) when is_list(Tokens) ->
    nif_forward_with_argmax(Ctx, Tokens).

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
nif_kv_pack(_Ctx, _Tokens, _NTokens, _SeqId) -> erlang:nif_error(nif_not_loaded).
nif_kv_unpack(_Ctx, _Bin, _SeqId) -> erlang:nif_error(nif_not_loaded).
nif_kv_seq_rm(_Ctx, _SeqId, _P0, _P1) -> erlang:nif_error(nif_not_loaded).
nif_apply_chat_template(_Model, _Request) -> erlang:nif_error(nif_not_loaded).
nif_embed(_Ctx, _Tokens) -> erlang:nif_error(nif_not_loaded).
nif_set_grammar(_Ctx, _Grammar) -> erlang:nif_error(nif_not_loaded).
nif_configure_sampler(_Ctx, _Cfg) -> erlang:nif_error(nif_not_loaded).
nif_clear_sampler(_Ctx) -> erlang:nif_error(nif_not_loaded).
nif_adapter_load(_Model, _Path) -> erlang:nif_error(nif_not_loaded).
nif_adapter_free(_Adapter) -> erlang:nif_error(nif_not_loaded).
nif_set_adapters(_Ctx, _Adapters) -> erlang:nif_error(nif_not_loaded).
nif_sampler_new(_Ctx, _Cfg) -> erlang:nif_error(nif_not_loaded).
nif_sampler_free(_Sampler) -> erlang:nif_error(nif_not_loaded).
nif_vram_info() -> erlang:nif_error(nif_not_loaded).
nif_model_size(_Model) -> erlang:nif_error(nif_not_loaded).
nif_model_n_layer(_Model) -> erlang:nif_error(nif_not_loaded).
nif_forward_with_argmax(_Ctx, _Tokens) -> erlang:nif_error(nif_not_loaded).
