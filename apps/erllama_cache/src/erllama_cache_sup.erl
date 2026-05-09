-module(erllama_cache_sup).
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init([]) ->
    SupFlags = #{strategy => one_for_one,
                 intensity => 5,
                 period => 30},
    %% Children are added incrementally per the implementation roadmap
    %% in plans/golden-finding-horizon.md. v0.1.0 ships an empty supervisor
    %% so the application starts cleanly.
    ChildSpecs = [],
    {ok, {SupFlags, ChildSpecs}}.
