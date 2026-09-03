# Hotword reading dictionary

The desktop Hotword editor embeds three compact exact-lookup tables. They are
input assistance only; the ASR server and decoder do not load them.

Sources used for the checked-in artifacts:

- SudachiDict Full 20260428 `system_full.dic`, SHA-256
  `2c993988aae44cbad92b395790c951aa2dad957c983a7d4c32944f6263e02593`
- CMUdict commit `74790861f652b15e4ac49015a90074ad62a27690`, `cmudict.dict`
  SHA-256
  `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`

First use the official `sudachi-cli dump <system_full.dic> winfo <winfo.csv>`
command. Then regenerate the embedded assets with:

```text
cargo run --release -p parapper-morph-dictionary \
  --bin build_hotword_reading_dictionary -- \
  <full-winfo.csv> <cmudict.dict> \
  src-tauri/resources/hotword-reading
```

The builder retains Kanji-containing Full surface forms as the Japanese baseline.
It also applies NFKC and lowercase normalization to Full, retaining exact ASCII
surface forms with kana readings. Sudachi duplicates are ranked by word cost and
limited to three readings. This makes entries such as
`GitHub -> ギットハブ` dictionary-derived without an application-specific override.
CMUdict retains up to two pronunciations per exact word for unknown-English
fallback. The runtime deliberately does not split CamelCase, hyphens, or other
unspaced compounds. It keeps the decompressed TSV plus 32-bit line offsets and
uses binary search, instead of expanding more than one million Kanji entries
into a `HashMap`.

The CMU fallback is intentionally a baseline, not a claim of canonical Japanese
spelling. It maps American-English phonemes to Japanese morae. At a word edge,
a voiceless stop or affricate after an ARPAbet lax vowel is geminated; voiced
obstruents and consonants after tense vowels are not. Word-internal gemination
is not guessed because its conditioning is more context-sensitive. Known
Japanese spellings from Sudachi Full always take precedence.

When both Sudachi Full and CMUdict miss an alphabetic word, the runtime offers
editable spelling-based candidates. The baseline handles common English chunks
(`tion`, `sion`, `ph`, `sh`, `ch`, `th`, `ck`, `qu`), vowel groups, silent final
`e`, soft `c`/`g`, intervocalic `s`, consonant codas, and short letter-name
candidates. It does not split CamelCase, hyphens, or underscore compounds.

These candidates are input assistance, not automatic canonical readings.
