%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_inflight).
-moduledoc """
Tracks in-flight streaming inference requests.

Each `erllama:infer/4` admit produces a unique `reference()` and an
entry in a public ETS table mapping that reference back to the
serving `erllama_model` gen_statem. `erllama:cancel/1` looks up the
ref here to find which model owns the request, then casts a
`{cancel, Ref}` event at it.

The table is a fixed-name public ETS so lookups are lock-free from
any process. The owning gen_server (this module) is here only to
keep the table alive across releases and to clean up entries when a
model dies unexpectedly.
""".
-behaviour(gen_server).

-export([start_link/0,
         register/2,
         unregister/1,
         lookup/1,
         all/0]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(TABLE, ?MODULE).

%% =============================================================================
%% Public API
%% =============================================================================

-spec register(reference(), pid()) -> ok.
register(Ref, ModelPid) when is_reference(Ref), is_pid(ModelPid) ->
    true = ets:insert(?TABLE, {Ref, ModelPid}),
    %% Monitor the model so a crash purges its inflight entries.
    %% The monitor is held by the gen_server, not by callers.
    gen_server:cast(?MODULE, {monitor_model, ModelPid}),
    ok.

-spec unregister(reference()) -> ok.
unregister(Ref) when is_reference(Ref) ->
    ets:delete(?TABLE, Ref),
    ok.

-spec lookup(reference()) -> {ok, pid()} | {error, not_found}.
lookup(Ref) when is_reference(Ref) ->
    case ets:lookup(?TABLE, Ref) of
        [{Ref, Pid}] -> {ok, Pid};
        []           -> {error, not_found}
    end.

-spec all() -> [{reference(), pid()}].
all() ->
    ets:tab2list(?TABLE).

%% =============================================================================
%% gen_server
%% =============================================================================

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    _ = ets:new(?TABLE, [named_table, public, set,
                         {read_concurrency, true},
                         {write_concurrency, true}]),
    {ok, #{monitors => #{}}}.

handle_call(_, _, S) -> {reply, {error, unknown_call}, S}.

handle_cast({monitor_model, Pid}, S = #{monitors := M}) ->
    case maps:is_key(Pid, M) of
        true  -> {noreply, S};
        false ->
            Ref = monitor(process, Pid),
            {noreply, S#{monitors := M#{Pid => Ref}}}
    end;
handle_cast(_, S) ->
    {noreply, S}.

handle_info({'DOWN', _MonRef, process, Pid, _Reason}, S = #{monitors := M}) ->
    %% Sweep all inflight entries owned by this dead model.
    Refs = [R || {R, P} <- ets:tab2list(?TABLE), P =:= Pid],
    [ets:delete(?TABLE, R) || R <- Refs],
    {noreply, S#{monitors := maps:remove(Pid, M)}};
handle_info(_, S) ->
    {noreply, S}.

terminate(_, _) -> ok.
