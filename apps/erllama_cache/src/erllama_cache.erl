%% @doc
%% Public API facade for the erllama_cache application.
%%
%% This module is the only public surface; tier servers, the meta
%% server, the writer pool, and the policy module are internal.
%%
%% v0.1.0 ships only the type and config exports; the runtime API
%% (`lookup_exact/1`, `checkout/2`, `checkin/1`, `save_async/4`,
%% `save_sync/4`, `lookup_exact_or_wait/2`) lands in step 5 of the
%% implementation roadmap (see `plans/golden-finding-horizon.md`).
%% @end
-module(erllama_cache).

-export_type([
    cache_key/0,
    tier/0,
    status/0,
    save_reason/0
]).

-type cache_key() :: <<_:256>>.
-type tier() :: ram | ram_file | disk.
-type status() :: available | writing | evicting.
-type save_reason() :: cold | continued | finish | evict | shutdown.
