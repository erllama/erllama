%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Deterministic stub backend for `erllama_model`.
%%
%% No NIF, no GGUF. tokenize uses `erlang:phash2/1` over whitespace-
%% delimited words; decode_one produces a deterministic next-token
%% from the context's hash; pack/unpack serialise the token list as
%% bytes. Useful for tests of the cache integration that don't need
%% real inference.
%% @end
-module(erllama_model_stub).
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
    seq_rm/2,
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
    thinking_signature/3,
    %% Test helpers: read back what the most recent configure_sampler
    %% / clear_sampler / apply_adapters call saw.
    last_sampler_cfg/1,
    cleared/1,
    applied_adapters/1
]).

%% Stub state.
%%
%% `sampler` is the currently-installed chain config (matches the real
%% backend's behaviour: it gets reset to #{} on clear_sampler/1).
%% `last_sampler` is the cfg the most recent configure_sampler/2 saw,
%% preserved across clear_sampler/1 so end-of-request cleanup doesn't
%% wipe out the value tests want to read.
-record(stub, {
    sampler = #{} :: map(),
    last_sampler = #{} :: map(),
    cleared = false :: boolean(),
    %% Most recently applied adapter set, as the {Ref, Scale} list the
    %% model layer passed to apply_adapters/2. Tests read this back to
    %% assert the snapshot rules.
    applied = [] :: [{reference(), float()}],
    %% Opt-in: when true, the first two decode ops per sampler emit
    %% `{thinking_token, _}`, the next emits `thinking_end`. Per-sampler
    %% phase is tracked via the process dictionary because step/2 takes
    %% no mutable state argument.
    thinking_capable = false :: boolean(),
    %% Opt-in: when true, two `{tool_call_token, _}` decodes followed
    %% by a `tool_call_end` are emitted after the thinking phase (or
    %% from the start when `thinking_capable = false`). Same per-sampler
    %% state machine as the thinking phase.
    tool_call_capable = false :: boolean(),
    %% Opt-in: extends the tool-call phase with payload markers,
    %% letting tests exercise the scheduler's sampler-swap on
    %% payload_open / payload_close. Only meaningful when
    %% tool_call_capable is also true.
    tool_call_payload_capable = false :: boolean()
}).

init(Config) ->
    {ok, #stub{
        thinking_capable = bool_opt(thinking_capable, Config),
        tool_call_capable = bool_opt(tool_call_capable, Config),
        tool_call_payload_capable = bool_opt(tool_call_payload_capable, Config)
    }}.

bool_opt(Key, Config) ->
    case maps:get(Key, Config, false) of
        true -> true;
        _ -> false
    end.

terminate(_S) ->
    ok.

tokenize(_S, Text) when is_binary(Text) ->
    [
        erlang:phash2(W) rem (1 bsl 32)
     || W <- binary:split(Text, <<" ">>, [global, trim_all]),
        W =/= <<>>
    ].

detokenize(_S, Tokens) ->
    list_to_binary(
        lists:join(<<" ">>, [integer_to_binary(T) || T <- Tokens])
    ).

prefill(_S, _Tokens) ->
    ok.

decode_one(_S, ContextTokens) ->
    Token = erlang:phash2({decode, ContextTokens}) rem (1 bsl 32),
    {ok, Token}.

kv_pack(_S, Tokens) ->
    erllama_cache_key:encode_tokens(Tokens).

kv_pack(_S, Tokens, _SeqId) ->
    erllama_cache_key:encode_tokens(Tokens).

kv_unpack(_S, _Bin) ->
    ok.

kv_unpack(_S, _Bin, _SeqId) ->
    ok.

%% Drops the per-seq phase counter when the scheduler releases a
%% seq's KV. Without this, a subsequent admission to the same
%% seq_id would inherit the prior request's state.
seq_rm(_S, SeqId) ->
    erlang:erase({stub_phase, SeqId}),
    ok.

%% Per-tick batched step. Prefill rows just acknowledge. Decode rows
%% derive a deterministic next token from
%% `{decode_step_stub, SeqId, Sampler}` so two concurrent seqs with
%% different prompts produce different streams and the same seq fed
%% the same sampler keeps producing the same token (matching the
%% prior single-seq stub semantics for cache-integration tests).
step(S, Ops) ->
    Results = [stub_step_op(Op, S) || Op <- Ops],
    {ok, Results}.

stub_step_op({SeqId, {prefill, _Tokens}}, _S) ->
    {SeqId, prefilled};
stub_step_op({SeqId, {decode, Sampler}}, #stub{
    thinking_capable = false,
    tool_call_capable = false
}) ->
    %% The token is bound to the sampler ref AND the seq_id so a
    %% scheduler bug that swaps samplers between seqs would change
    %% one seq's output stream and be visible in tests. The mock
    %% never produces eog.
    decode_token(SeqId, Sampler);
stub_step_op(
    {SeqId, {decode, Sampler}},
    #stub{
        thinking_capable = TC,
        tool_call_capable = UC,
        tool_call_payload_capable = PC
    }
) ->
    Phase = next_phase(SeqId, TC, UC),
    advance_phase(SeqId, Sampler, Phase, PC).

%% Phases (per-seq_id, via process dict) walk in order:
%%   thinking_emit_0, thinking_emit_1, thinking_done,
%%   tool_call_emit_0, [payload_open, payload_emit, payload_close,]
%%   tool_call_emit_1 (when no payload markers), tool_call_end_due,
%%   normal
%% Phases for disabled features are skipped, so a stub with only
%% thinking_capable goes thinking_emit_0 -> thinking_emit_1 ->
%% thinking_done -> normal. The key is seq_id (not sampler ref)
%% so mid-request sampler swaps (the greedy-on-syntax path) don't
%% reset the state machine.
next_phase(SeqId, TC, UC) ->
    case erlang:get({stub_phase, SeqId}) of
        undefined when TC -> thinking_emit_0;
        undefined when UC -> tool_call_emit_0;
        undefined -> normal;
        thinking_done when UC -> tool_call_emit_0;
        thinking_done -> normal;
        Other -> Other
    end.

%% Phase transitions: each clause picks the next phase, then either
%% emits a token (via with_phase_token/4) or a bare marker (via
%% with_phase/3).
advance_phase(SeqId, Sampler, thinking_emit_0, _PC) ->
    with_phase_token(SeqId, Sampler, thinking_emit_1, {thinking, 0});
advance_phase(SeqId, Sampler, thinking_emit_1, _PC) ->
    with_phase_token(SeqId, Sampler, thinking_end_due, {thinking, 1});
advance_phase(SeqId, _Sampler, thinking_end_due, _PC) ->
    with_phase(SeqId, thinking_done, thinking_end);
advance_phase(SeqId, Sampler, tool_call_emit_0, true) ->
    with_phase_token(SeqId, Sampler, tool_call_payload_open_due, {tool_call, 0});
advance_phase(SeqId, Sampler, tool_call_emit_0, false) ->
    with_phase_token(SeqId, Sampler, tool_call_emit_1, {tool_call, 0});
advance_phase(SeqId, Sampler, tool_call_emit_1, _PC) ->
    with_phase_token(SeqId, Sampler, tool_call_end_due, {tool_call, 1});
advance_phase(SeqId, Sampler, tool_call_payload_open_due, _PC) ->
    with_phase_marker(SeqId, Sampler, tool_call_payload_emit, payload_open);
advance_phase(SeqId, Sampler, tool_call_payload_emit, _PC) ->
    with_phase_token(SeqId, Sampler, tool_call_payload_close_due, {tool_call, payload_body});
advance_phase(SeqId, Sampler, tool_call_payload_close_due, _PC) ->
    with_phase_marker(SeqId, Sampler, tool_call_end_due, payload_close);
advance_phase(SeqId, _Sampler, tool_call_end_due, _PC) ->
    with_phase(SeqId, normal, tool_call_end);
advance_phase(SeqId, Sampler, normal, _PC) ->
    decode_token(SeqId, Sampler).

%% Advance the per-seq phase and return a step_result with a token
%% derived from the (seq, sampler, seed) tuple. The seed
%% disambiguates ticks within the same phase so two consecutive
%% thinking_tokens get distinct ids.
with_phase_token(SeqId, Sampler, NextPhase, Seed) ->
    with_phase_tagged(SeqId, Sampler, NextPhase, Seed, tag_for_seed(Seed)).

with_phase_marker(SeqId, Sampler, NextPhase, Marker) ->
    with_phase_tagged(SeqId, Sampler, NextPhase, Marker, tag_for_marker(Marker)).

with_phase_tagged(SeqId, Sampler, NextPhase, Seed, Tag) ->
    erlang:put({stub_phase, SeqId}, NextPhase),
    T = erlang:phash2({tool_call_step_stub, SeqId, Sampler, Seed}) rem (1 bsl 32),
    {SeqId, {Tag, T}}.

tag_for_seed({thinking, _}) -> thinking_token;
tag_for_seed({tool_call, _}) -> tool_call_token.

tag_for_marker(payload_open) -> tool_call_payload_open;
tag_for_marker(payload_close) -> tool_call_payload_close.

with_phase(SeqId, NextPhase, MarkerResult) ->
    erlang:put({stub_phase, SeqId}, NextPhase),
    {SeqId, MarkerResult}.

decode_token(SeqId, Sampler) ->
    T = erlang:phash2({decode_step_stub, SeqId, Sampler}) rem (1 bsl 32),
    {SeqId, {token, T, 0}}.

%% Deterministic per-seq stub signature. The stub ignores `Bytes`
%% and hashes the seq_id; real backends derive their signature from
%% the observed thinking text via HMAC.
thinking_signature(_S, SeqId, _Bytes) ->
    crypto:hash(sha256, <<"stub-thinking-sig-", (integer_to_binary(SeqId))/binary>>).

%% Sampler refs are opaque references. Free drops the per-sampler
%% per-sampler stub phase (a no-op when neither thinking_capable nor
%% tool_call_capable is set).
sampler_new(_S, _Cfg) ->
    {ok, make_ref()}.

sampler_free(_Sampler) ->
    %% Stub state is now keyed on seq_id; cleanup happens in seq_rm.
    ok.

%% Render a chat request as `system\nrole: content\nrole: content\n`
%% and tokenise via the same phash2 scheme as tokenize/2. Tools are
%% inlined into the system prefix. Deterministic and roundtrippable
%% enough for tests.
apply_chat_template(S, Request) when is_map(Request) ->
    Messages = maps:get(messages, Request, []),
    System = maps:get(system, Request, undefined),
    Tools = maps:get(tools, Request, undefined),
    Rendered = render(System, Tools, Messages),
    {ok, tokenize(S, Rendered)}.

%% A 16-dim hash-derived embedding vector. Deterministic per token
%% list. Useful for /v1/embeddings shape testing without a real model.
embed(_S, Tokens) when is_list(Tokens) ->
    Seed = erlang:phash2({embed, Tokens}),
    Vec = [
        float((Seed bsr (I * 4)) band 16#FFFF) / 65535.0
     || I <- lists:seq(0, 15)
    ],
    {ok, Vec}.

%% Stub backend doesn't sample (decode_one returns a phash2-derived
%% token deterministically), so grammar / sampler params are ignored.
%% The most recent config is recorded on the state so tests can read
%% it back via `last_sampler_cfg/1`.
set_grammar(#stub{sampler = Cfg} = S, Grammar) when is_binary(Grammar) ->
    NewCfg = Cfg#{grammar => Grammar},
    {ok, S#stub{sampler = NewCfg, last_sampler = NewCfg, cleared = false}};
set_grammar(#stub{} = S, undefined) ->
    {ok, S}.

configure_sampler(#stub{} = S, Cfg) when is_map(Cfg) ->
    {ok, S#stub{sampler = Cfg, last_sampler = Cfg, cleared = false}}.

clear_sampler(#stub{} = S) ->
    {ok, S#stub{sampler = #{}, cleared = true}}.

last_sampler_cfg(#stub{last_sampler = Cfg}) -> Cfg.
cleared(#stub{cleared = C}) -> C.
applied_adapters(#stub{applied = A}) -> A.

%% LoRA stubs: the adapter handle is a fresh reference per load so
%% tests can distinguish multiple adapters; unload is a no-op;
%% apply_adapters just records the call.
load_adapter(#stub{} = S, _Path) ->
    {ok, make_ref(), S}.

unload_adapter(#stub{} = S, _Ref) ->
    {ok, S}.

apply_adapters(#stub{} = S, Adapters) when is_list(Adapters) ->
    {ok, S#stub{applied = Adapters}}.

%% =============================================================================
%% Internal: chat-template rendering
%% =============================================================================

render(System, Tools, Messages) ->
    Header =
        case System of
            undefined -> [];
            <<>> -> [];
            _ -> [<<"system: ">>, System, <<"\n">>]
        end,
    ToolsBlob = render_tools(Tools),
    Body = [render_message(M) || M <- Messages],
    iolist_to_binary([Header, ToolsBlob, Body]).

render_tools(undefined) ->
    [];
render_tools([]) ->
    [];
render_tools(Tools) when is_list(Tools) ->
    Lines = [render_tool(T) || T <- Tools],
    [<<"tools:\n">>, Lines].

render_tool(#{name := Name} = T) ->
    Desc = maps:get(description, T, <<>>),
    [<<"  - ">>, Name, <<": ">>, Desc, <<"\n">>].

render_message(#{role := Role, content := Content}) when is_binary(Content) ->
    [Role, <<": ">>, Content, <<"\n">>];
render_message(#{role := Role, content := Blocks}) when is_list(Blocks) ->
    Texts = [B || #{type := <<"text">>, text := B} <- Blocks],
    [Role, <<": ">>, lists:join(<<" ">>, Texts), <<"\n">>];
render_message(_) ->
    <<>>.
