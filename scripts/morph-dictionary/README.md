# Built-in morph dictionary

Parapper embeds a compact UniDic CWJ 3.1.1 dictionary in the application with
`include_bytes!`. Morph boundary detection therefore works without a separate
dictionary download or installation step.

The dictionary uses Vibrato's Raw connector representation
(`dual_connector = false`). Source rows are reduced to `surface`, left context
ID, right context ID, word cost, and a four-digit feature. The feature format
is `[PP][S][F]`: two digits for the primary part of speech, one for the
boundary-relevant subtype, and one for the conjugation form. Other UniDic
feature columns are not included.

## Rebuilding

Generate `src-tauri/resources/morph/system.dic.zst` from the UniDic CWJ 3.1.1
source files with:

```powershell
cargo run --release --locked -p parapper-morph-dictionary --bin build_morph_dictionary -- `
  <builder-input>/lex.csv `
  <builder-input>/feature.def `
  <builder-input>/right-id.def `
  <builder-input>/left-id.def `
  <builder-input>/model.def `
  <builder-input>/dicrc `
  <builder-input>/char.def `
  <builder-input>/unk.def `
  src-tauri/resources/morph/system.dic.zst
```

Keep the generated dictionary together with `AUTHORS`, `BSD`, and `NOTICE` in
`src-tauri/resources/morph/`. The build bundles these notices into the
installer, and the application exposes them under Settings > Licenses.

The separate hotword reading dictionaries and their generation procedure are
documented in [HOTWORD_READING_DICTIONARY.md](HOTWORD_READING_DICTIONARY.md).
