%% @doc
%% Pure-Erlang policy decisions for the erllama_cache subsystem.
%%
%% Two responsibilities:
%%
%%   1. Boundary trim: cold saves persist a *trimmed-aligned prefix* of
%%      the prompt rather than the full live token list, so the next
%%      request whose prompt is a textual extension of this one still
%%      lands on the saved cache key after BPE retokenisation. The
%%      algorithm trims a fixed number of tokens off the tail and
%%      aligns the result down to a multiple of a configured chunk.
%%
%%   2. Save-reason gating: cold/continued/finish saves each have a
%%      simple guard (token-count thresholds and intervals). Eviction
%%      and shutdown saves are unconditional and do not pass through
%%      this module.
%%
%% This module has no side effects; everything is testable as plain
%% data transformations.
%% @end
-module(erllama_cache_policy).

-export([
    trim_boundary/3,
    cold_save_split/2,
    should_continued_save/3,
    should_finish_save/2,
    validate_config/1
]).

-export_type([config/0, token/0]).

-type token() :: non_neg_integer().

-type config() :: #{
    min_tokens := non_neg_integer(),
    cold_min_tokens := non_neg_integer(),
    cold_max_tokens := non_neg_integer(),
    continued_interval := pos_integer(),
    boundary_trim_tokens := non_neg_integer(),
    boundary_align_tokens := pos_integer()
}.

%% =============================================================================
%% Boundary trim
%% =============================================================================

-spec trim_boundary([token()], non_neg_integer(), pos_integer()) ->
    {ok, [token()]} | {skip, too_short}.
trim_boundary(Tokens, Trim, Align) when
    is_list(Tokens), is_integer(Trim), Trim >= 0, is_integer(Align), Align > 0
->
    Len = length(Tokens),
    case trim_count(Len, Trim, Align) of
        {ok, N} -> {ok, lists:sublist(Tokens, N)};
        {skip, Reason} -> {skip, Reason}
    end.

-spec trim_count(non_neg_integer(), non_neg_integer(), pos_integer()) ->
    {ok, non_neg_integer()} | {skip, too_short}.
trim_count(Len, Trim, Align) ->
    AfterTrim = Len - Trim,
    case AfterTrim < Align of
        true -> {skip, too_short};
        false -> {ok, (AfterTrim div Align) * Align}
    end.

%% =============================================================================
%% Save-reason gating
%% =============================================================================

%% Decide whether a cold save fires, and if so, return both the trimmed
%% prefix to pack/save and the remaining tokens still to be prefilled
%% into the live context.
-spec cold_save_split([token()], config()) ->
    {trim, [token()], [token()]} | no_save.
cold_save_split(Tokens, Cfg) ->
    Len = length(Tokens),
    Min = maps:get(cold_min_tokens, Cfg),
    Max = maps:get(cold_max_tokens, Cfg),
    Trim = maps:get(boundary_trim_tokens, Cfg),
    Align = maps:get(boundary_align_tokens, Cfg),
    case Len < Min orelse Len > Max of
        true ->
            no_save;
        false ->
            case trim_count(Len, Trim, Align) of
                {ok, N} ->
                    {Prefix, Rest} = lists:split(N, Tokens),
                    {trim, Prefix, Rest};
                {skip, _} ->
                    no_save
            end
    end.

%% Continued saves fire every `continued_interval` tokens of *new*
%% generation (i.e. live token count minus the count at the last save).
-spec should_continued_save(non_neg_integer(), non_neg_integer(), config()) ->
    boolean().
should_continued_save(LiveCount, LastSavedAtCount, Cfg) when
    is_integer(LiveCount),
    LiveCount >= 0,
    is_integer(LastSavedAtCount),
    LastSavedAtCount >= 0
->
    Interval = maps:get(continued_interval, Cfg),
    Min = maps:get(min_tokens, Cfg),
    LiveCount - LastSavedAtCount >= Interval andalso LiveCount >= Min.

%% Finish saves fire at successful end-of-stream provided the live
%% sequence is at or above the global minimum.
-spec should_finish_save(non_neg_integer(), config()) -> boolean().
should_finish_save(LiveCount, Cfg) when is_integer(LiveCount), LiveCount >= 0 ->
    LiveCount >= maps:get(min_tokens, Cfg).

%% =============================================================================
%% Config validation
%% =============================================================================

-spec validate_config(map()) -> ok | {error, term()}.
validate_config(Cfg) ->
    Required = [
        min_tokens,
        cold_min_tokens,
        cold_max_tokens,
        continued_interval,
        boundary_trim_tokens,
        boundary_align_tokens
    ],
    case [K || K <- Required, not maps:is_key(K, Cfg)] of
        [] -> check_invariants(Cfg);
        Missing -> {error, {missing_keys, Missing}}
    end.

-spec check_invariants(config()) -> ok | {error, term()}.
check_invariants(Cfg) ->
    Min = maps:get(min_tokens, Cfg),
    ColdMin = maps:get(cold_min_tokens, Cfg),
    ColdMax = maps:get(cold_max_tokens, Cfg),
    Interval = maps:get(continued_interval, Cfg),
    Trim = maps:get(boundary_trim_tokens, Cfg),
    Align = maps:get(boundary_align_tokens, Cfg),
    Checks = [
        {is_integer(Min) andalso Min >= 0, {invalid, min_tokens, Min}},
        {is_integer(ColdMin) andalso ColdMin >= Min, {ordering, cold_min_tokens_lt_min_tokens}},
        {
            is_integer(ColdMax) andalso ColdMax >= ColdMin,
            {ordering, cold_max_tokens_lt_cold_min_tokens}
        },
        {is_integer(Interval) andalso Interval > 0, {invalid, continued_interval, Interval}},
        {is_integer(Trim) andalso Trim >= 0, {invalid, boundary_trim_tokens, Trim}},
        {is_integer(Align) andalso Align > 0, {invalid, boundary_align_tokens, Align}}
    ],
    case lists:dropwhile(fun({Pass, _}) -> Pass end, Checks) of
        [] -> ok;
        [{_, Reason} | _] -> {error, Reason}
    end.
