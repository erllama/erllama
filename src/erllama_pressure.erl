%% @doc
%% Behaviour and helpers for memory-pressure samplers used by
%% `erllama_scheduler`. A sampler is a stateless module that returns
%% the current `{Used, Total}` byte tuple for the resource it tracks
%% (system RAM, GPU VRAM, or a custom source).
%%
%% A sampler MUST be cheap. The scheduler invokes `sample/0` on every
%% tick (default 5 s); slow samplers will block the scheduler's mailbox.
%% If a sampler needs to spawn a port (e.g. `nvidia-smi`), it should
%% cache the previous reading and return it when the call would block,
%% or run its own background poller.
%% @end
-module(erllama_pressure).

-export([sample/1, available_sources/0]).

-export_type([source/0, reading/0]).

-type source() ::
    noop
    | system
    | nvidia_smi
    | {module, module()}.
-type reading() :: {non_neg_integer(), non_neg_integer()}.

-callback sample() -> reading().

-spec sample(source()) -> reading().
sample(noop) ->
    {0, 1};
sample(system) ->
    erllama_pressure_system:sample();
sample(nvidia_smi) ->
    erllama_pressure_nvidia_smi:sample();
sample({module, Mod}) when is_atom(Mod) ->
    Mod:sample().

-spec available_sources() -> [source()].
available_sources() ->
    [noop, system, nvidia_smi].
