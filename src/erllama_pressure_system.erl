%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_pressure_system).
-moduledoc """
System-memory pressure sampler backed by OTP's `memsup` (from
`os_mon`). Portable across Linux, macOS, BSD, and Windows. Returns
`{Total - Available, Total}`.

Requires `os_mon` to be started. The scheduler ensures this when
`system` is selected as the pressure source.
""".
-behaviour(erllama_pressure).

-export([sample/0]).

-spec sample() -> erllama_pressure:reading().
sample() ->
    Data = memsup:get_system_memory_data(),
    Total = pick(Data, [system_total_memory, total_memory], 0),
    Avail = pick(Data, [available_memory, free_memory], 0),
    case Total of
        0 -> {0, 1};
        _ -> {max(Total - Avail, 0), Total}
    end.

pick(_Data, [], Default) ->
    Default;
pick(Data, [Key | Rest], Default) ->
    case proplists:get_value(Key, Data) of
        undefined -> pick(Data, Rest, Default);
        N when is_integer(N), N > 0 -> N;
        _ -> pick(Data, Rest, Default)
    end.
