# Parapper compact morph dictionary

## Decision

The Parapper native and future WASM dictionaries both use Vibrato's Raw connector representation (`dual_connector = false`). The source UniDic rows are reduced to the four dictionary columns (`surface`, left context ID, right context ID, word cost) and one four-digit feature before dictionary construction. The feature format is `[PP][S][F]`: two digits for the primary part of speech, one for the boundary-relevant subtype, and one for the conjugation form. No other UniDic feature columns are stored.

Native development uses the published package through Parapper's model download flow. A valid installation contains both:

```text
%APPDATA%\com.parakeet-inc.parapper\models\unidic-cwj-3_1_1\system.dic
%APPDATA%\com.parakeet-inc.parapper\models\unidic-cwj-3_1_1\manifest.json
```

Copying only `system.dic` is unsupported because the runtime uses the install
manifest to bind the representation, feature encoding, expanded size, and
SHA-256 verified during installation. Pre-release generated artifacts should
be exercised through the ignored real-dictionary test using
`PARAPPER_MORPH_ARTIFACT_DIR`; they should not be copied into the model
directory as a partial installation.

Do not silently fall back to the old full-feature dictionary. The runtime can parse its legacy feature strings for migration compatibility, but the installed v0.4.0 dictionary is expected to contain only four ASCII digits per token.

## Generated artifacts

The 2026-07-16 build from UniDic CWJ 3.1.1 produced:

| Artifact | Representation | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `system.dic.zst` | native rkyv, compact Raw | 7,469,299 | `92fbc19982ada2d565115819d11083d51dd1285c4dfa059112fafcad79bcca86` |
| expanded `system.dic` | native rkyv, compact Raw | 44,497,344 | `1d7f0a194ae1f296f740fdb08433ef1fce81c0a353e7ce3585146b295ca87ef6` |

`AUTHORS` and `BSD` from the UniDic source package, plus the derivative
`NOTICE`, must accompany the dictionary assets. This follows UniDic's
published BSD redistribution guidance for software packages. The packaged
`AUTHORS` is fixed to the official 34-byte form without a trailing newline
(SHA-256 `d3cde34eb1ec1f6b4b4d78b77d2ff909f812a3c71d451cf97adb48ce7b1b0ab5`).

WASM is not a v0.4.0 Release asset. Its build and release contract are deferred
to v0.5.0, where it will be regenerated and revalidated rather than reusing a
provisional v0.4.0-era artifact.

The v0.4.0 builder generates only the native dictionary:

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
  <output>/system.dic.zst
```

The provisional WASM build path and its tests are intentionally omitted until
the v0.5.0 representation and release contract are defined.

## Release contract before v0.4.0

Publish a dedicated dictionary Release from the Parapper-ASR release branch immediately before the Parapper v0.4.0 Release. The immutable contract is:

| Field | Value |
| --- | --- |
| repository | `Parakeet-Inc/Parapper-ASR` |
| tag | `morph-dictionary-unidic-cwj-3.1.1-v1` |
| native asset | `parapper-unidic-cwj-3_1_1-compact-raw-v1.tar.zst` |
| attribution and license assets | `AUTHORS`, `BSD`, `NOTICE` |
| metadata assets | `release-manifest.json`, `SHA256SUMS` |

The native package contains exactly:

```text
unidic-cwj-3_1_1/
  AUTHORS
  BSD
  NOTICE
  system.dic.zst
  manifest.json
  SHA256SUMS
```

`AUTHORS`, `BSD`, and `NOTICE` are top-level Release assets as well as native
package members, making the source attribution, license, and derivative notice
directly visible on the Release.
Both metadata levels bind every payload to an exact byte count and SHA-256.
The package manifest additionally records the expanded native dictionary
identity, `compact-raw` representation, and `[PP][S][F]` feature encoding.
Extra, missing, duplicated, truncated, or modified content is rejected.

Stage and verify the assets without publishing:

```powershell
./scripts/morph-dictionary/release-morph-dictionary.ps1 `
  -SourceDirectory <generated-artifact-directory> `
  -OutputDirectory artifacts/morph-dictionary-release
```

The source directory must contain `system.dic.zst` (or the audited expanded
`system.dic`), `AUTHORS`, `BSD`, and `NOTICE`. A WASM dictionary in that directory is
ignored and is never added to the v0.4.0 Release. If only `system.dic` is
present, the script validates its fixed 44,497,344-byte / SHA-256 identity and
passes those already serialized rkyv bytes through the builder's Rust zstd
level-19 path. This reproduces `system.dic.zst` at 7,469,299 bytes with SHA-256
`92fbc19982ada2d565115819d11083d51dd1285c4dfa059112fafcad79bcca86`.

After reviewing the generated `release-manifest.json`, publish the fixed Release:

```powershell
./scripts/morph-dictionary/release-morph-dictionary.ps1 `
  -SourceDirectory <generated-artifact-directory> `
  -OutputDirectory artifacts/morph-dictionary-release `
  -Publish
```

Without `-Publish`, the script refuses a non-empty output directory. With
`-Publish`, it intentionally reuses and re-verifies the already reviewed
output directory. It creates or resumes a draft Release, uploads all six
assets, downloads them to a fresh temporary directory, and runs the full
verification again before publishing the draft. A failed upload or downloaded
verification leaves the Release unpublished for inspection and retry. The
dictionary-only Release is explicitly excluded from GitHub's Latest release.
Publication is never an implicit side effect of packaging.
The Release tag is created from the public repository's `main` branch by default;
use `-Target` only when the intended public commit already exists in
`Parakeet-Inc/Parapper-ASR`.

After the asset package name and Release tag are fixed, record the immutable URL, exact package byte count, and package SHA-256 in the application. On first use, download to a temporary file, reject a size or SHA-256 mismatch, materialize `system.dic`, verify the rkyv magic and expected expanded size, and only then atomically install it. Until that Release exists, use only the local generated artifact; do not point development builds at a mutable or provisional URL.
