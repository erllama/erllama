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
    kv_unpack/2,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    %% Test helpers: read back what the most recent configure_sampler
    %% / clear_sampler call saw.
    last_sampler_cfg/1,
    cleared/1
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
    cleared = false :: boolean()
}).

init(_Config) ->
    {ok, #stub{}}.

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

kv_unpack(_S, _Bin) ->
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
