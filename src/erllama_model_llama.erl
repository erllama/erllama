%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_llama).
-moduledoc """
Real-llama.cpp backend for `erllama_model`.

Owns a `model_ref` and a `context_ref` from `erllama_nif`. The
gen_statem hands its decode/kv operations through this module;
this module forwards to the NIF.

Config (passed through `erllama_model:start_link/2`):
  model_path :: file:name() | binary()  (required)
  model_opts :: map()  (forwarded to erllama_nif:load_model/2)
  context_opts :: map()  (forwarded to erllama_nif:new_context/2)

`model_opts` and `context_opts` flow through to the NIF unchanged.
See `erllama_nif:load_model/2` and `erllama_nif:new_context/2` for
the full set of recognised keys, including the llama.cpp option
passthroughs `split_mode`, `main_gpu`, `tensor_split`,
`flash_attn`, `type_k`, and `type_v`.
""".
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    detokenize/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_pack/3,
    kv_unpack/2,
    kv_unpack/3,
    seq_clear/1,
    seq_rm/2,
    seq_rm_last/2,
    seq_rm_last/3,
    step/2,
    sampler_new/2,
    sampler_free/1,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    load_adapter/2,
    unload_adapter/2,
    apply_adapters/2,
    extra_metadata/1,
    verify/4,
    thinking_signature/3
]).

-record(s, {
    model :: erllama_nif:model_ref(),
    ctx :: erllama_nif:context_ref(),
    %% Captured once at init for vram_estimate_b derivation. The
    %% values are immutable once the model is loaded; the gen_statem
    %% reads them via extra_metadata/1 at its own init time and
    %% caches the derived estimate in #data.
    model_size_bytes = 0 :: non_neg_integer(),
    total_layers = 0 :: non_neg_integer(),
    %% Signed: llama.cpp uses negative (typically -1) to mean
    %% "offload all layers".
    n_gpu_layers = 0 :: integer(),
    %% Token ids that open / close a thinking block. Populated at
    %% init from the optional `thinking_markers` config key by
    %% tokenising the start / end strings through this model's own
    %% vocabulary. Empty lists disable marker recognition entirely;
    %% the backend then behaves identically to a non-thinking
    %% backend regardless of the per-request thinking flag.
    thinking_start_ids = [] :: [erllama_nif:token_id()],
    thinking_end_ids = [] :: [erllama_nif:token_id()],
    %% Same shape for tool-call spans. Populated from the optional
    %% `tool_call_markers` config key. Empty lists disable
    %% recognition; the backend then behaves identically to a
    %% non-tool-call backend.
    tool_call_start_ids = [] :: [erllama_nif:token_id()],
    tool_call_end_ids = [] :: [erllama_nif:token_id()],
    %% Inner payload markers inside a tool-call span. When set the
    %% scheduler switches the request's sampler off the greedy
    %% variant for tokens between them, so caller-supplied string
    %% arguments stay diverse while syntax stays byte-deterministic.
    %% Empty lists leave the whole span on the greedy sampler.
    tool_call_payload_start_ids = [] :: [erllama_nif:token_id()],
    tool_call_payload_end_ids = [] :: [erllama_nif:token_id()]
}).

init(Config) ->
    Path = maps:get(model_path, Config),
    MOpts = maps:get(model_opts, Config, #{}),
    case erllama_nif:load_model(Path, MOpts) of
        {ok, Model} -> open_context(Model, Config, MOpts);
        {error, _} = E -> E
    end.

open_context(Model, Config, MOpts) ->
    COpts = maps:get(context_opts, Config, #{}),
    case erllama_nif:new_context(Model, COpts) of
        {ok, Ctx} ->
            {ok, build_state(Model, Ctx, Config, MOpts)};
        {error, _} = E ->
            erllama_nif:free_model(Model),
            E
    end.

build_state(Model, Ctx, Config, MOpts) ->
    ThinkingMarkers = maps:get(thinking_markers, Config, #{}),
    ToolCallMarkers = maps:get(tool_call_markers, Config, #{}),
    {ThinkStart, ThinkEnd} = tokenize_markers(Model, ThinkingMarkers),
    {ToolStart, ToolEnd} = tokenize_markers(Model, ToolCallMarkers),
    {PayStart, PayEnd} = tokenize_payload_markers(Model, ToolCallMarkers),
    #s{
        model = Model,
        ctx = Ctx,
        model_size_bytes = safe_uint(erllama_nif:model_size(Model)),
        total_layers = safe_uint(erllama_nif:model_n_layer(Model)),
        n_gpu_layers = maps:get(n_gpu_layers, MOpts, 0),
        thinking_start_ids = ThinkStart,
        thinking_end_ids = ThinkEnd,
        tool_call_start_ids = ToolStart,
        tool_call_end_ids = ToolEnd,
        tool_call_payload_start_ids = PayStart,
        tool_call_payload_end_ids = PayEnd
    }.

safe_uint(N) when is_integer(N), N >= 0 -> N;
safe_uint(_) -> 0.

%% Resolve the configured `thinking_markers => #{start => Binary,
%% end => Binary}` map into two token-id lists by running each
%% string through the model's own vocabulary. Missing or invalid
%% entries yield an empty list, which keeps the step wrapper a
%% no-op for that side.
tokenize_markers(_Model, Markers) when map_size(Markers) =:= 0 ->
    {[], []};
tokenize_markers(Model, Markers) when is_map(Markers) ->
    Start = tokenize_marker(Model, maps:get(start, Markers, undefined)),
    End = tokenize_marker(Model, maps:get('end', Markers, undefined)),
    {Start, End};
tokenize_markers(_Model, _Other) ->
    {[], []}.

tokenize_marker(_Model, undefined) ->
    [];
tokenize_marker(_Model, <<>>) ->
    [];
tokenize_marker(Model, Bin) when is_binary(Bin) ->
    case erllama_nif:tokenize(Model, Bin, #{add_special => false, parse_special => true}) of
        Tokens when is_list(Tokens) -> Tokens;
        _ -> []
    end.

%% Tool-call payload markers live under the same `tool_call_markers`
%% map but are optional. When both are absent the whole tool-call
%% span stays on the greedy sampler.
tokenize_payload_markers(_Model, Markers) when map_size(Markers) =:= 0 ->
    {[], []};
tokenize_payload_markers(Model, Markers) when is_map(Markers) ->
    Start = tokenize_marker(Model, maps:get(payload_start, Markers, undefined)),
    End = tokenize_marker(Model, maps:get(payload_end, Markers, undefined)),
    {Start, End};
tokenize_payload_markers(_Model, _Other) ->
    {[], []}.

terminate(#s{ctx = Ctx, model = Model}) ->
    erllama_nif:free_context(Ctx),
    erllama_nif:free_model(Model),
    ok.

tokenize(#s{model = M}, Text) ->
    erllama_nif:tokenize(M, Text, #{add_special => true, parse_special => false}).

detokenize(#s{model = M}, Tokens) ->
    erllama_nif:detokenize(M, Tokens).

prefill(#s{ctx = C}, Tokens) ->
    erllama_nif:prefill(C, Tokens).

decode_one(#s{ctx = C}, _ContextTokens) ->
    erllama_nif:decode_one(C).

kv_pack(#s{ctx = C}, Tokens) ->
    erllama_nif:kv_pack(C, Tokens, length(Tokens)).

%% Seq-aware kv_pack. The 2-arity above keeps the v0.1 contract
%% (seq_id=0); the 3-arity is driven by the multi-sequence scheduler.
kv_pack(#s{ctx = C}, Tokens, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_pack(C, Tokens, length(Tokens), SeqId).

kv_unpack(#s{ctx = C}, Bin) ->
    erllama_nif:kv_unpack(C, Bin, 0).

kv_unpack(#s{ctx = C}, Bin, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_unpack(C, Bin, SeqId).

%% Drop the cell at position N-1 from seq 0 so the model layer can
%% re-prefill the corresponding token and regenerate logits.
%% `llama_state_seq_*` only persists KV cells, never the per-context
%% logits buffer; without this primer the next sample would read stale
%% (or zero) logits.
seq_rm_last(#s{ctx = C}, NTokens) when NTokens > 0 ->
    erllama_nif:kv_seq_rm(C, 0, NTokens - 1, -1).

%% Seq-aware variant of seq_rm_last/2.
seq_rm_last(#s{ctx = C}, SeqId, NTokens) when
    is_integer(SeqId), SeqId >= 0, NTokens > 0
->
    erllama_nif:kv_seq_rm(C, SeqId, NTokens - 1, -1).

%% Free all KV cells of a specific seq_id. Used by the scheduler when
%% a request finishes and its seq_id returns to the idle pool.
seq_rm(#s{ctx = C}, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_seq_rm(C, SeqId, 0, -1).

%% Wipe seq 0 entirely. p0=0, p1=-1 means "from position 0 to infinity".
seq_clear(#s{ctx = C}) ->
    erllama_nif:kv_seq_rm(C, 0, 0, -1).

%% Drive one batched-decode tick. Forwards to the NIF and, when
%% thinking or tool-call markers are configured, maps any sampled
%% token that matches one of them into the corresponding step-result
%% variant. With every marker list empty the mapping is the
%% identity and the return shape is exactly what the NIF produced.
%% Thinking checks come first so a token id that is in both sets
%% routes to thinking.
step(#s{ctx = C} = S, Ops) ->
    case erllama_nif:step(C, Ops) of
        {ok, Results} ->
            case all_markers_empty(S) of
                true -> {ok, Results};
                false -> {ok, [map_marker(R, S) || R <- Results]}
            end;
        Other ->
            Other
    end.

all_markers_empty(#s{
    thinking_start_ids = [],
    thinking_end_ids = [],
    tool_call_start_ids = [],
    tool_call_end_ids = [],
    tool_call_payload_start_ids = [],
    tool_call_payload_end_ids = []
}) ->
    true;
all_markers_empty(_) ->
    false.

map_marker({SeqId, {token, Tok, _Eog}} = R, #s{
    thinking_start_ids = TSI,
    thinking_end_ids = TEI,
    tool_call_start_ids = USI,
    tool_call_end_ids = UEI,
    tool_call_payload_start_ids = PSI,
    tool_call_payload_end_ids = PEI
}) ->
    Membership = {
        lists:member(Tok, TSI),
        lists:member(Tok, TEI),
        lists:member(Tok, USI),
        lists:member(Tok, UEI),
        lists:member(Tok, PSI),
        lists:member(Tok, PEI)
    },
    case Membership of
        {true, _, _, _, _, _} -> {SeqId, {thinking_token, Tok}};
        {_, true, _, _, _, _} -> {SeqId, thinking_end};
        {_, _, true, _, _, _} -> {SeqId, {tool_call_token, Tok}};
        {_, _, _, true, _, _} -> {SeqId, tool_call_end};
        {_, _, _, _, true, _} -> {SeqId, {tool_call_payload_open, Tok}};
        {_, _, _, _, _, true} -> {SeqId, {tool_call_payload_close, Tok}};
        _ -> R
    end;
map_marker(R, _S) ->
    R.

%% Build a per-request sampler chain. The opaque sampler_ref is held
%% by the scheduler for the request's lifetime and freed when the
%% request finishes.
sampler_new(#s{ctx = C}, Cfg) when is_map(Cfg) ->
    erllama_nif:sampler_new(C, Cfg).

sampler_free(SamplerRef) ->
    erllama_nif:sampler_free(SamplerRef).

apply_chat_template(#s{model = M}, Request) ->
    erllama_nif:apply_chat_template(M, Request).

embed(#s{ctx = C}, Tokens) ->
    erllama_nif:embed(C, Tokens).

set_grammar(#s{ctx = C} = S, Grammar) when is_binary(Grammar) ->
    case erllama_nif:set_grammar(C, Grammar) of
        ok -> {ok, S};
        {error, _} = E -> E
    end;
set_grammar(#s{} = S, undefined) ->
    {ok, S}.

configure_sampler(#s{} = S, Cfg) when map_size(Cfg) =:= 0 ->
    %% No sampler params at all - leave the existing chain alone so
    %% the lazy greedy fallback in the NIF kicks in on first decode.
    {ok, S};
configure_sampler(#s{ctx = C} = S, Cfg) when is_map(Cfg) ->
    case erllama_nif:configure_sampler(C, Cfg) of
        ok -> {ok, S};
        {error, _} = E -> E
    end.

clear_sampler(#s{ctx = C} = S) ->
    ok = erllama_nif:clear_sampler(C),
    {ok, S}.

load_adapter(#s{model = M} = S, Path) ->
    case erllama_nif:adapter_load(M, Path) of
        {ok, AdapterRef} -> {ok, AdapterRef, S};
        {error, _} = E -> E
    end.

unload_adapter(#s{} = S, AdapterRef) ->
    case erllama_nif:adapter_free(AdapterRef) of
        ok -> {ok, S};
        {error, released} -> {ok, S};
        {error, _} = E -> E
    end.

apply_adapters(#s{ctx = C} = S, Adapters) ->
    case erllama_nif:set_adapters(C, Adapters) of
        ok -> {ok, S};
        {error, _} = E -> E
    end.

extra_metadata(#s{
    model_size_bytes = SB, total_layers = TL, n_gpu_layers = NL
}) ->
    #{model_size_bytes => SB, total_layers => TL, n_gpu_layers => NL}.

%% Speculative-decoding verifier. Snapshot+restore protocol so
%% the caller's KV view is unchanged after the call. Empty prefix
%% is rejected: the acceptance / NextToken indexing both require
%% at least one prefix token.
verify(_S, [], _Candidates, _K) ->
    {error, empty_prefix};
verify(#s{ctx = Ctx} = S, PrefixTokens, Candidates, K) when
    is_list(PrefixTokens), is_list(Candidates), is_integer(K), K > 0
->
    KCap = min(K, length(Candidates)),
    Truncated = lists:sublist(Candidates, KCap),
    Input = PrefixTokens ++ Truncated,
    case erllama_nif:forward_with_argmax(Ctx, Input) of
        {ok, Argmax} ->
            P = length(PrefixTokens),
            Accepted = count_accepted(0, P, Truncated, Argmax),
            %% NextToken comes from logits at 0-indexed position
            %% P - 1 + Accepted (the verifier's prediction given
            %% the prefix and the accepted prefix of candidates).
            %% lists:nth/2 is 1-indexed: P + Accepted.
            NextToken = lists:nth(P + Accepted, Argmax),
            ok = restore_after_verify(Ctx, P, lists:last(PrefixTokens)),
            {ok, Accepted, NextToken, S};
        {error, _} = E ->
            E
    end.

count_accepted(Acc, _P, [], _Argmax) ->
    Acc;
count_accepted(Acc, P, [C | Rest], Argmax) ->
    Predicted = lists:nth(P + Acc, Argmax),
    case Predicted of
        C when is_integer(C) -> count_accepted(Acc + 1, P, Rest, Argmax);
        _ -> Acc
    end.

%% Bring the context's KV cells back to the caller's pre-call
%% length P, then re-prefill the last prefix token so the logits
%% buffer is in a sampleable state for any follow-up decode_one.
%% The caller's pre-call decode_ready flag is not preserved: after
%% verify the context is always ready to sample at the prefix end.
%% Documented in the public erllama:verify/4 doc.
restore_after_verify(Ctx, P, LastPrefixToken) ->
    %% kv_seq_rm with p1 = -1 means "to infinity"; combined with
    %% p0 = P, that drops every cell from the candidate batch.
    _ = erllama_nif:kv_seq_rm(Ctx, 0, P, -1),
    _ = erllama_nif:prefill(Ctx, [LastPrefixToken]),
    ok.

%% Sign the observed thinking-phase bytes with an HMAC-SHA256 over
%% the node-wide signing key. The Anthropic SDK round-trips the
%% resulting binary as `signature_delta` so the server can verify
%% the thinking text on the next turn. The key is read from the
%% application environment:
%%
%%     application:set_env(erllama, thinking_signing_key, <<"...">>).
%%
%% Leaving the key unset returns `<<>>`, the documented
%% "no signature available" path: the downstream omits
%% `signature_delta` from its SSE output.
thinking_signature(_S, _SeqId, Bytes) when is_binary(Bytes) ->
    case application:get_env(erllama, thinking_signing_key) of
        {ok, Key} when is_binary(Key), Key =/= <<>> ->
            crypto:mac(hmac, sha256, Key, Bytes);
        _ ->
            <<>>
    end.
