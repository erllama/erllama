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
    tool_call_capable = false :: boolean()
}).

init(Config) ->
    {ok, #stub{
        thinking_capable = bool_opt(thinking_capable, Config),
        tool_call_capable = bool_opt(tool_call_capable, Config)
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

%% No persistent per-seq state in the stub, so seq_rm is a no-op.
%% The scheduler's tests rely on this being callable; the cache
%% integration goes through the encoded-token binary instead.
seq_rm(_S, _SeqId) ->
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
    #stub{thinking_capable = TC, tool_call_capable = UC}
) ->
    Phase = next_phase(Sampler, TC, UC),
    advance_phase(Sampler, Phase, SeqId).

%% Phases (per-sampler, via process dict) walk in order:
%%   thinking_emit_0, thinking_emit_1, thinking_done,
%%   tool_call_emit_0, tool_call_emit_1, tool_call_done,
%%   normal
%% Phases for disabled features are skipped, so a stub with only
%% thinking_capable goes thinking_emit_0 -> thinking_emit_1 ->
%% thinking_done -> normal.
next_phase(Sampler, TC, UC) ->
    case erlang:get({stub_phase, Sampler}) of
        undefined when TC -> thinking_emit_0;
        undefined when UC -> tool_call_emit_0;
        undefined -> normal;
        thinking_done when UC -> tool_call_emit_0;
        thinking_done -> normal;
        Other -> Other
    end.

advance_phase(Sampler, thinking_emit_0, SeqId) ->
    erlang:put({stub_phase, Sampler}, thinking_emit_1),
    T = erlang:phash2({thinking_step_stub, SeqId, Sampler, 0}) rem (1 bsl 32),
    {SeqId, {thinking_token, T}};
advance_phase(Sampler, thinking_emit_1, SeqId) ->
    erlang:put({stub_phase, Sampler}, thinking_end_due),
    T = erlang:phash2({thinking_step_stub, SeqId, Sampler, 1}) rem (1 bsl 32),
    {SeqId, {thinking_token, T}};
advance_phase(Sampler, thinking_end_due, SeqId) ->
    erlang:put({stub_phase, Sampler}, thinking_done),
    {SeqId, thinking_end};
advance_phase(Sampler, tool_call_emit_0, SeqId) ->
    erlang:put({stub_phase, Sampler}, tool_call_emit_1),
    T = erlang:phash2({tool_call_step_stub, SeqId, Sampler, 0}) rem (1 bsl 32),
    {SeqId, {tool_call_token, T}};
advance_phase(Sampler, tool_call_emit_1, SeqId) ->
    erlang:put({stub_phase, Sampler}, tool_call_end_due),
    T = erlang:phash2({tool_call_step_stub, SeqId, Sampler, 1}) rem (1 bsl 32),
    {SeqId, {tool_call_token, T}};
advance_phase(Sampler, tool_call_end_due, SeqId) ->
    erlang:put({stub_phase, Sampler}, normal),
    {SeqId, tool_call_end};
advance_phase(Sampler, normal, SeqId) ->
    decode_token(SeqId, Sampler).

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

sampler_free(Sampler) ->
    erlang:erase({stub_phase, Sampler}),
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
