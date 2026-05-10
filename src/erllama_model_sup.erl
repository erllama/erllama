%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_sup).
-moduledoc """
Dynamic supervisor for `erllama_model` gen_statems. Each loaded
model is one child started via `start_model/2`. simple_one_for_one
strategy: children are spawned on demand from a single child spec.
""".
-behaviour(supervisor).

-export([start_link/0, start_model/2, stop_model/1, models/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

-doc """
Start a model under this supervisor. ModelId is a binary; the model
registers itself in `erllama_registry` under that key. Returns the
started pid.
""".
-spec start_model(binary(), map()) -> {ok, pid()} | {error, term()}.
start_model(ModelId, Config) when is_binary(ModelId), is_map(Config) ->
    supervisor:start_child(?SERVER, [ModelId, Config]).

-doc "Terminate a previously started model.".
-spec stop_model(binary() | pid()) -> ok | {error, term()}.
stop_model(ModelOrPid) ->
    case pid_of(ModelOrPid) of
        undefined -> {error, not_found};
        Pid       -> supervisor:terminate_child(?SERVER, Pid)
    end.

-doc "List currently-supervised model pids (raw supervisor view).".
-spec models() -> [{undefined, pid(), worker, [module()]}].
models() ->
    supervisor:which_children(?SERVER).

init([]) ->
    SupFlags = #{
        strategy => simple_one_for_one,
        intensity => 5,
        period => 30
    },
    Child = #{
        id => erllama_model,
        start => {erllama_model, start_link, []},
        restart => transient,
        shutdown => 5000,
        type => worker,
        modules => [erllama_model]
    },
    {ok, {SupFlags, [Child]}}.

pid_of(Pid) when is_pid(Pid) -> Pid;
pid_of(ModelId) when is_binary(ModelId) -> erllama_registry:whereis_name(ModelId).
