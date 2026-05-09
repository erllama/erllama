-module(prop_erllama_cache_policy).
-include_lib("proper/include/proper.hrl").

%% =============================================================================
%% trim_boundary/3 properties
%% =============================================================================

%% The returned prefix length is always a multiple of Align.
prop_trimmed_length_is_aligned() ->
    ?FORALL(
        {Tokens, Trim, Align},
        {list(non_neg_integer()), non_neg_integer(), pos_integer()},
        case erllama_cache_policy:trim_boundary(Tokens, Trim, Align) of
            {ok, Prefix} ->
                length(Prefix) rem Align =:= 0;
            {skip, too_short} ->
                length(Tokens) - Trim < Align
        end
    ).

%% A successful trim returns a strict-or-equal prefix of the input.
prop_trimmed_is_prefix() ->
    ?FORALL(
        {Tokens, Trim, Align},
        {list(non_neg_integer()), non_neg_integer(), pos_integer()},
        case erllama_cache_policy:trim_boundary(Tokens, Trim, Align) of
            {ok, Prefix} ->
                Prefix =:= lists:sublist(Tokens, length(Prefix));
            {skip, _} ->
                true
        end
    ).

%% Trim never extends the input.
prop_trimmed_within_input() ->
    ?FORALL(
        {Tokens, Trim, Align},
        {list(non_neg_integer()), non_neg_integer(), pos_integer()},
        case erllama_cache_policy:trim_boundary(Tokens, Trim, Align) of
            {ok, Prefix} -> length(Prefix) =< length(Tokens);
            {skip, _} -> true
        end
    ).

%% A successful trim leaves at least `Trim` tokens off the tail (the
%% whole point of the algorithm).
prop_trimmed_drops_at_least_trim() ->
    ?FORALL(
        {Tokens, Trim, Align},
        {list(non_neg_integer()), non_neg_integer(), pos_integer()},
        case erllama_cache_policy:trim_boundary(Tokens, Trim, Align) of
            {ok, Prefix} -> length(Tokens) - length(Prefix) >= Trim;
            {skip, _} -> true
        end
    ).

%% =============================================================================
%% cold_save_split/2 properties
%% =============================================================================

valid_cfg() ->
    ?LET(
        {Min, ColdGap, ColdSpan, Interval, Trim, Align},
        {
            range(0, 1024),
            range(0, 1024),
            range(0, 60000),
            pos_integer(),
            non_neg_integer(),
            pos_integer()
        },
        #{
            min_tokens => Min,
            cold_min_tokens => Min + ColdGap,
            cold_max_tokens => Min + ColdGap + ColdSpan,
            continued_interval => Interval,
            boundary_trim_tokens => Trim,
            boundary_align_tokens => Align
        }
    ).

%% The returned prefix and rest concatenate back to the original input.
prop_cold_split_concat_equals_input() ->
    ?FORALL(
        {Tokens, Cfg},
        {list(non_neg_integer()), valid_cfg()},
        case erllama_cache_policy:cold_save_split(Tokens, Cfg) of
            {trim, Prefix, Rest} -> Prefix ++ Rest =:= Tokens;
            no_save -> true
        end
    ).

%% A {trim, Prefix, _} result implies the prefix length is a multiple
%% of Align AND in [cold_min_tokens, cold_max_tokens] for the input.
prop_cold_split_respects_bounds() ->
    ?FORALL(
        {Tokens, Cfg},
        {list(non_neg_integer()), valid_cfg()},
        case erllama_cache_policy:cold_save_split(Tokens, Cfg) of
            {trim, Prefix, _} ->
                Len = length(Tokens),
                Align = maps:get(boundary_align_tokens, Cfg),
                length(Prefix) rem Align =:= 0 andalso
                    Len >= maps:get(cold_min_tokens, Cfg) andalso
                    Len =< maps:get(cold_max_tokens, Cfg);
            no_save ->
                true
        end
    ).

%% =============================================================================
%% validate_config/1 property
%% =============================================================================

prop_validate_config_accepts_valid_cfg() ->
    ?FORALL(
        Cfg,
        valid_cfg(),
        ok =:= erllama_cache_policy:validate_config(Cfg)
    ).
