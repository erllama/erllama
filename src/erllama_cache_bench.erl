%% @doc
%% Microbench helpers for the cache subsystem.
%%
%% These do NOT measure realistic prefill / decode latency — that
%% requires the real `erllama_nif` against llama.cpp (step 2b).
%% What they do measure: framing / CRC / link-publish / iommap-load
%% latency. Useful as a regression guard on the I/O path and as a
%% template for the post-2b benchmark that will assert the >=10x
%% cold-vs-warm speedup target on NVMe.
%%
%% Usage from the shell:
%%
%%   1> application:ensure_all_started(erllama).
%%   2> {ok, _} = erllama_cache_disk_srv:start_link(b_disk, "/tmp/b").
%%   3> erllama_cache_bench:save_load(b_disk, 100, 4096).
%% @end
-module(erllama_cache_bench).

-export([save_load/3]).

-spec save_load(atom(), pos_integer(), pos_integer()) ->
    #{
        save_us_avg := non_neg_integer(),
        load_us_avg := non_neg_integer(),
        runs := pos_integer(),
        payload_bytes := pos_integer()
    }.
save_load(DiskSrv, Runs, PayloadBytes) when
    is_atom(DiskSrv),
    is_integer(Runs),
    Runs > 0,
    is_integer(PayloadBytes),
    PayloadBytes > 0
->
    Payload = binary:copy(<<"x">>, PayloadBytes),
    SaveMicros = bench_loop(Runs, fun(I) -> bench_save(DiskSrv, I, Payload) end),
    LoadMicros = bench_loop(Runs, fun(I) -> bench_load(DiskSrv, I) end),
    #{
        save_us_avg => SaveMicros div Runs,
        load_us_avg => LoadMicros div Runs,
        runs => Runs,
        payload_bytes => PayloadBytes
    }.

bench_loop(Runs, Fun) ->
    lists:foldl(
        fun(I, Acc) ->
            T0 = erlang:monotonic_time(microsecond),
            _ = Fun(I),
            T1 = erlang:monotonic_time(microsecond),
            Acc + (T1 - T0)
        end,
        0,
        lists:seq(1, Runs)
    ).

bench_save(DiskSrv, I, Payload) ->
    Tokens = [I],
    Meta = #{
        save_reason => cold,
        quant_bits => 16,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => Tokens,
        context_size => 4096,
        prompt_text => <<>>,
        hostname => <<"bench">>,
        erllama_version => <<"0.1.0">>
    },
    erllama_cache_disk_srv:save(DiskSrv, Meta, Payload).

bench_load(DiskSrv, I) ->
    Key = erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AA>>, 32),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => [I]
    }),
    erllama_cache_disk_srv:load(DiskSrv, Key).
