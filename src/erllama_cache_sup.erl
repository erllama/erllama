-module(erllama_cache_sup).
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init([]) ->
    SupFlags = #{strategy => one_for_one, intensity => 5, period => 30},
    Children = [erllama_cache_meta_srv, erllama_cache_ram, erllama_cache_writer],
    {ok, {SupFlags, [worker(M) || M <- Children]}}.

worker(Mod) ->
    #{
        id => Mod,
        start => {Mod, start_link, []},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [Mod]
    }.
