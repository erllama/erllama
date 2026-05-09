%% @doc
%% NVIDIA GPU memory-pressure sampler. Aggregates VRAM usage across
%% every GPU on the host via `nvidia-smi --query-gpu=memory.used,
%% memory.total --format=csv,noheader,nounits`.
%%
%% Returns `{TotalUsedBytes, TotalCapacityBytes}` summed over all GPUs.
%% A host with no GPU or where `nvidia-smi` is missing reports
%% `{0, 1}` so the scheduler treats it as zero pressure.
%% @end
-module(erllama_pressure_nvidia_smi).
-behaviour(erllama_pressure).

-export([sample/0]).

-define(CMD,
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null"
).

-spec sample() -> erllama_pressure:reading().
sample() ->
    case os:cmd(?CMD) of
        [] ->
            {0, 1};
        Out ->
            Lines = [string:trim(L) || L <- string:split(Out, "\n", all), L =/= []],
            {U, T} = lists:foldl(fun fold_line/2, {0, 0}, Lines),
            case T of
                0 -> {0, 1};
                _ -> {U * 1024 * 1024, T * 1024 * 1024}
            end
    end.

fold_line(Line, {U, T}) ->
    case string:split(Line, ",") of
        [UsedS, TotalS] ->
            case {parse_int(UsedS), parse_int(TotalS)} of
                {{ok, Used}, {ok, Total}} -> {U + Used, T + Total};
                _ -> {U, T}
            end;
        _ ->
            {U, T}
    end.

parse_int(S) ->
    case string:to_integer(string:trim(S)) of
        {N, _} when is_integer(N) -> {ok, N};
        _ -> error
    end.
