# Models4me

One CLI that lists **every model you can actually reach** — the `opencode` catalog
unioned with every model your own API keys unlock — and talks to either set.

Most tools show you one side. `opencode models` knows the models opencode ships with;
your API keys know a different, often much larger set, including gateways opencode
never lists. Models4me holds both, labels each row with where it came from, and can
send a prompt through either route.

## Why both lists

Measured on one machine with a normal set of stored keys:

```
opencode alone  :  1,104
keys alone      :  7,755
shared          :    920   (83% of opencode's list is confirmed by a key)
union           :  7,939   = the keys' 7,755 + 184 models only opencode has
```

Neither side is a superset of the other: the keys add 6,835 rows opencode does not
list, and opencode adds 184 rows the keys cannot reach (providers whose `/models`
endpoint is templated or refuses to enumerate). The union is the point.

## Install

No packaging step, no extra dependencies beyond what the tool imports:

```bash
pip install rich httpx
curl -O https://raw.githubusercontent.com/Alexander-Sorrell-IT/models4me/main/models4me.py
python3 models4me.py list
```

`opencode` is optional but recommended — without it you only see the key side.
Python 3.8+.

## Commands

```
models4me                      list models (default command)
models4me list -p nvidia       one provider
models4me list -s both         only models both sides agree on
models4me providers            endpoints, key source, model counts
models4me diff nvidia          the exact IDs opencode and your key disagree on
models4me probe -p groq        test models with a real request
models4me ask groq/openai/gpt-oss-20b "say pong"
models4me show <model>         everything known about one model
models4me use <#|id>           remember a default
models4me extraFuff            Freebuff's models, levels and per-model criteria
```

Every command takes `--json`. Run `models4me COMMAND --help` for options.

### The `From` column

| tag | meaning |
| --- | --- |
| `both` | opencode lists it **and** the provider returned it for your key |
| `key` | only the provider's own `/models` returned it for your key |
| `oc` | only opencode lists it (the provider did not confirm it) |

`--source opencode|direct|both|key` filters on that.

### `extraFuff`

Reads Freebuff's published model table and plans page, then shows each model's
access level, plan requirement and what Freebuff says it is for — plus how many
IDs in your own catalog match it, so a model you like there is one `ask` away.
Add `-v` for the full criteria per model. Freebuff has no public model API, so
the data comes from its README and `/plans`, cached for a day (`--refresh` re-reads).

## Where credentials come from

Models4me **never stores, writes or transmits your keys anywhere except to the
provider they belong to.** Resolution order per provider:

1. the process environment (`NVIDIA_API_KEY`, `GROQ_API_KEY`, …)
2. keys already held by [omnicli](https://www.npmjs.com/package/omnicli) —
   `~/.config/omnicli/config.json` `keys`, its `keys.env`, `~/.config/trio/keys.env`
3. whatever `opencode auth export` reports

This repository contains no keys of any kind, and the program has no code path
that writes one to disk. Nothing here imports omnicli's code either — it only
reads the keys that already exist on the machine.

## Caches

`~/.cache/models4me/` (override with `MODELS4ME_CACHE`): provider table from
models.dev, the opencode model list, each provider's `/models` answer (24 h),
probe verdicts, and the Freebuff pages. Default model lives in
`~/.config/models4me/config.json` (`MODELS4ME_CONFIG`).

## License

MIT License. See [LICENSE](LICENSE) for details.
