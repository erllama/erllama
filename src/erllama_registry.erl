%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_registry).
-moduledoc """
ETS-backed `via` callback for naming `erllama_model` gen_statems by
binary `model_id()`.

Atoms are unbounded global table entries; user-supplied model
identifiers (e.g. coming from an HTTP request body in a front-end
server) cannot safely be `binary_to_atom/1`'d. This registry lets a
gen_statem be registered as

```
gen_statem:start_link({via, erllama_registry, ModelId}, ...)
```

with `ModelId :: binary()`, and looked up via `whereis_name/1`.

The registry is a tiny gen_server that owns a public ETS table.
Lookups happen straight from ETS in the caller process, so the
gen_server is never on the hot path.
""".
-behaviour(gen_server).

-export([
    start_link/0,
    register_name/2,
    unregister_name/1,
    whereis_name/1,
    send/2,
    all/0
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(TABLE, ?MODULE).

-type model_id() :: binary().
-export_type([model_id/0]).

%% =============================================================================
%% via API
%% =============================================================================

-spec register_name(model_id(), pid()) -> yes | no.
register_name(Name, Pid) when is_binary(Name), is_pid(Pid) ->
    case ets:insert_new(?TABLE, {Name, Pid, monitor(process, Pid)}) of
        true -> yes;
        false -> no
    end.

-spec unregister_name(model_id()) -> model_id().
unregister_name(Name) when is_binary(Name) ->
    case ets:lookup(?TABLE, Name) of
        [{Name, _Pid, MonRef}] ->
            demonitor(MonRef, [flush]),
            ets:delete(?TABLE, Name);
        [] ->
            ok
    end,
    Name.

-spec whereis_name(model_id()) -> pid() | undefined.
whereis_name(Name) when is_binary(Name) ->
    case ets:lookup(?TABLE, Name) of
        [{Name, Pid, _}] when is_pid(Pid) ->
            case is_process_alive(Pid) of
                true -> Pid;
                false -> undefined
            end;
        [] ->
            undefined
    end.

-spec send(model_id(), term()) -> pid().
send(Name, Msg) when is_binary(Name) ->
    case whereis_name(Name) of
        Pid when is_pid(Pid) ->
            Pid ! Msg,
            Pid;
        undefined ->
            error({badarg, {Name, Msg}})
    end.

%% Snapshot of all currently-registered models. Used by
%% erllama:list_models/0.
-spec all() -> [{model_id(), pid()}].
all() ->
    [
        {Name, Pid}
     || {Name, Pid, _Mon} <- ets:tab2list(?TABLE),
        is_process_alive(Pid)
    ].

%% =============================================================================
%% gen_server
%% =============================================================================

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    _ = ets:new(?TABLE, [
        named_table,
        public,
        set,
        {read_concurrency, true},
        {write_concurrency, true}
    ]),
    {ok, #{}}.

handle_call(_, _, S) -> {reply, {error, unknown_call}, S}.
handle_cast(_, S) -> {noreply, S}.

handle_info({'DOWN', _MonRef, process, Pid, _Reason}, S) ->
    Entries = ets:match_object(?TABLE, {'_', Pid, '_'}),
    lists:foreach(fun({Name, _, _}) -> ets:delete(?TABLE, Name) end, Entries),
    {noreply, S};
handle_info(_, S) ->
    {noreply, S}.

terminate(_, _) -> ok.
