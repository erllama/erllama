-module(erllama_app).
-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
    ok = erllama_cache_counters:init(),
    erllama_sup:start_link().

stop(_State) ->
    ok.
