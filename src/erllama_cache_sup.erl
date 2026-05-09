-module(erllama_cache_sup).
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init([]) ->
    SupFlags = #{
        strategy => one_for_one,
        intensity => 5,
        period => 30
    },
    ChildSpecs = [
        #{
            id => erllama_cache_meta_srv,
            start => {erllama_cache_meta_srv, start_link, []},
            restart => permanent,
            shutdown => 5000,
            type => worker,
            modules => [erllama_cache_meta_srv]
        },
        #{
            id => erllama_cache_ram,
            start => {erllama_cache_ram, start_link, []},
            restart => permanent,
            shutdown => 5000,
            type => worker,
            modules => [erllama_cache_ram]
        }
    ],
    {ok, {SupFlags, ChildSpecs}}.
