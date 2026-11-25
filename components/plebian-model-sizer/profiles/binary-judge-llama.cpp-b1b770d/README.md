# llama.cpp v1.0.0 binary-judge profile

Measured on 2026-11-24 for `qwen3.5-0.8b` Q8_0
(`37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`) on the
host class `x86_64-i7-9850h-12t-62g`. The engine is llama.cpp v1.0.0,
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, CPU backend, built by
`kilix-llm build-runtime.sh` with `GGML_NATIVE=ON` and no patches.

`measure.py` loaded the base GGUF alone, with no adapter, at 512, 2048, 8192,
32768, 131072 and 262144 tokens. Each point is `VmHWM` of `llama-server`
after one completion whose prompt fills the context (context minus 8 tokens,
one token out) using the judge's server flags. The stand-in adapter's
21,671,008 bytes were added to each peak, and to the real judge proof's peak,
because that file is not on this machine. `prove.py` then ran one real
`kilix-llm judge` at the trained-context limit (261,965 input tokens). That
peak sits on the line, so its headroom is 0.

From `components/plebian-model-sizer`, with `SERVER` the kilix-llm
`llama-server` and `MODEL` the Q8_0 GGUF:

```sh
D=profiles/binary-judge-llama.cpp-b1b770d
M=$D/qwen3.5-0.8b-q8_0.cpu.x86_64-i7-9850h-12t-62g.json
python3 $D/measure.py "$SERVER" "$MODEL" "$M"
uv run --locked --offline python $D/fit.py "$M" 178
python3 $D/prove.py /path/to/kilix-llm "$PWD/plebian-model-sizer" \
  $D/judge-proof.x86_64-i7-9850h-12t-62g.json
uv run --locked --offline python $D/fit.py "$M" 178 \
  $D/judge-proof.x86_64-i7-9850h-12t-62g.json
```

`fit.py` needs the locked environment. System Python does not have
`plebian_hardware`. The 262144-token point and the proof each took about
four hours on this host.
