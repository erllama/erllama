%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_policy_tests).
-include_lib("eunit/include/eunit.hrl").

cfg() ->
    #{
        min_tokens => 512,
        cold_min_tokens => 512,
        cold_max_tokens => 30000,
        continued_interval => 2048,
        boundary_trim_tokens => 32,
        boundary_align_tokens => 2048
    }.

%% =============================================================================
%% trim_boundary/3
%% =============================================================================

trim_boundary_too_short_returns_skip_test() ->
    %% Len=600, Trim=32, Align=2048 -> AfterTrim=568 < 2048
    ?assertEqual(
        {skip, too_short},
        erllama_cache_policy:trim_boundary(lists:seq(1, 600), 32, 2048)
    ).

trim_boundary_at_align_threshold_test() ->
    %% Len=2080, Trim=32 -> AfterTrim=2048, Align=2048
    Tokens = lists:seq(1, 2080),
    {ok, Prefix} = erllama_cache_policy:trim_boundary(Tokens, 32, 2048),
    ?assertEqual(2048, length(Prefix)),
    ?assertEqual(lists:sublist(Tokens, 2048), Prefix).

trim_boundary_aligns_down_test() ->
    %% Len=4500, Trim=32 -> AfterTrim=4468; (4468 div 2048) * 2048 = 4096
    Tokens = lists:seq(1, 4500),
    {ok, Prefix} = erllama_cache_policy:trim_boundary(Tokens, 32, 2048),
    ?assertEqual(4096, length(Prefix)).

trim_boundary_zero_trim_test() ->
    Tokens = lists:seq(1, 4096),
    {ok, Prefix} = erllama_cache_policy:trim_boundary(Tokens, 0, 2048),
    ?assertEqual(4096, length(Prefix)).

trim_boundary_align_one_is_identity_test() ->
    Tokens = lists:seq(1, 100),
    %% Trim=0, Align=1 -> always full length
    {ok, Prefix} = erllama_cache_policy:trim_boundary(Tokens, 0, 1),
    ?assertEqual(Tokens, Prefix).

trim_boundary_trim_exceeds_length_test() ->
    %% Trim larger than Len: AfterTrim is negative, < Align -> too_short
    ?assertEqual(
        {skip, too_short},
        erllama_cache_policy:trim_boundary(lists:seq(1, 10), 100, 2048)
    ).

trim_boundary_empty_input_test() ->
    ?assertEqual(
        {skip, too_short},
        erllama_cache_policy:trim_boundary([], 0, 1)
    ).

trim_boundary_badarg_negative_trim_test() ->
    ?assertError(function_clause, erllama_cache_policy:trim_boundary([1, 2, 3], -1, 1)).

trim_boundary_badarg_zero_align_test() ->
    ?assertError(function_clause, erllama_cache_policy:trim_boundary([1, 2, 3], 0, 0)).

%% =============================================================================
%% cold_save_split/2
%% =============================================================================

cold_save_split_below_min_test() ->
    Tokens = lists:seq(1, 400),
    ?assertEqual(no_save, erllama_cache_policy:cold_save_split(Tokens, cfg())).

cold_save_split_above_max_test() ->
    %% cold_max_tokens=30000; use 30001 tokens
    Tokens = lists:seq(1, 30001),
    ?assertEqual(no_save, erllama_cache_policy:cold_save_split(Tokens, cfg())).

cold_save_split_in_range_but_unaligned_test() ->
    %% Len=600, in [512..30000] but trim=32 leaves 568 < 2048 -> no_save
    Tokens = lists:seq(1, 600),
    ?assertEqual(no_save, erllama_cache_policy:cold_save_split(Tokens, cfg())).

cold_save_split_returns_prefix_and_rest_test() ->
    Tokens = lists:seq(1, 4500),
    {trim, Prefix, Rest} = erllama_cache_policy:cold_save_split(Tokens, cfg()),
    ?assertEqual(4096, length(Prefix)),
    ?assertEqual(404, length(Rest)),
    ?assertEqual(Tokens, Prefix ++ Rest).

cold_save_split_at_lower_bound_test() ->
    %% Len=2080: passes lower bound (>=512), trim=32 -> aligned 2048
    Tokens = lists:seq(1, 2080),
    {trim, Prefix, Rest} = erllama_cache_policy:cold_save_split(Tokens, cfg()),
    ?assertEqual(2048, length(Prefix)),
    ?assertEqual(32, length(Rest)).

%% =============================================================================
%% should_continued_save/3
%% =============================================================================

should_continued_save_below_interval_test() ->
    %% Generated 1000 tokens since last save, interval=2048
    ?assertNot(erllama_cache_policy:should_continued_save(3048, 2048, cfg())).

should_continued_save_at_interval_test() ->
    ?assert(erllama_cache_policy:should_continued_save(4096, 2048, cfg())).

should_continued_save_below_min_total_test() ->
    %% Even though delta >= interval, total live count is below min_tokens
    ?assertNot(erllama_cache_policy:should_continued_save(400, 0, cfg())).

%% =============================================================================
%% should_finish_save/2
%% =============================================================================

should_finish_save_below_min_test() ->
    ?assertNot(erllama_cache_policy:should_finish_save(400, cfg())).

should_finish_save_at_min_test() ->
    ?assert(erllama_cache_policy:should_finish_save(512, cfg())).

should_finish_save_above_min_test() ->
    ?assert(erllama_cache_policy:should_finish_save(8192, cfg())).

%% =============================================================================
%% validate_config/1
%% =============================================================================

validate_config_ok_test() ->
    ?assertEqual(ok, erllama_cache_policy:validate_config(cfg())).

validate_config_missing_key_test() ->
    Bad = maps:remove(continued_interval, cfg()),
    ?assertMatch(
        {error, {missing_keys, [continued_interval]}},
        erllama_cache_policy:validate_config(Bad)
    ).

validate_config_cold_min_below_min_test() ->
    Cfg = cfg(),
    Bad = Cfg#{cold_min_tokens => 100},
    ?assertMatch(
        {error, {ordering, cold_min_tokens_lt_min_tokens}},
        erllama_cache_policy:validate_config(Bad)
    ).

validate_config_cold_max_below_cold_min_test() ->
    Cfg = cfg(),
    Bad = Cfg#{cold_max_tokens => 100},
    ?assertMatch(
        {error, {ordering, cold_max_tokens_lt_cold_min_tokens}},
        erllama_cache_policy:validate_config(Bad)
    ).

validate_config_zero_align_test() ->
    Cfg = cfg(),
    Bad = Cfg#{boundary_align_tokens => 0},
    ?assertMatch(
        {error, {invalid, boundary_align_tokens, 0}},
        erllama_cache_policy:validate_config(Bad)
    ).

validate_config_zero_interval_test() ->
    Cfg = cfg(),
    Bad = Cfg#{continued_interval => 0},
    ?assertMatch(
        {error, {invalid, continued_interval, 0}},
        erllama_cache_policy:validate_config(Bad)
    ).
