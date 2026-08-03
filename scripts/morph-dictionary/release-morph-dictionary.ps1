param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,
    [string]$OutputDirectory = "artifacts/morph-dictionary-release",
    [string]$Repository = "Parakeet-Inc/Parapper-ASR",
    [string]$Target = "main",
    [switch]$Publish
)

$ErrorActionPreference = "Stop"
$releaseTag = "morph-dictionary-unidic-cwj-3.1.1-v1"
$nativeAsset = "parapper-unidic-cwj-3_1_1-compact-raw-v1.tar.zst"

$source = [System.IO.Path]::GetFullPath($SourceDirectory)
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$compressed = Join-Path $source "system.dic.zst"
$expanded = Join-Path $source "system.dic"
$outputHasAssets = (Test-Path -LiteralPath $output -PathType Container) -and
    (@(Get-ChildItem -LiteralPath $output -Force).Count -ne 0)

if ($outputHasAssets) {
    if (-not $Publish) {
        throw "Release output directory must be empty unless -Publish reuses verified assets: $output"
    }
} else {
    if (-not (Test-Path -LiteralPath $compressed -PathType Leaf)) {
        if (-not (Test-Path -LiteralPath $expanded -PathType Leaf)) {
            throw "Neither system.dic.zst nor system.dic exists in $source"
        }
        cargo run --release --locked -p parapper-morph-dictionary --bin release_morph_dictionary -- `
            prepare-native $expanded $compressed
        if ($LASTEXITCODE -ne 0) {
            throw "Native dictionary preparation failed."
        }
    }

    cargo run --release --locked -p parapper-morph-dictionary --bin release_morph_dictionary -- `
        package $source $output
    if ($LASTEXITCODE -ne 0) {
        throw "Dictionary packaging failed."
    }
}

cargo run --release --locked -p parapper-morph-dictionary --bin release_morph_dictionary -- `
    verify $output
if ($LASTEXITCODE -ne 0) {
    throw "Dictionary Release verification failed."
}

Get-Content -LiteralPath (Join-Path $output "release-manifest.json")

if (-not $Publish) {
    Write-Host "Verified local assets. Re-run with -Publish to create $releaseTag in $Repository."
    exit 0
}

$existingReleaseJson = gh release view $releaseTag --repo $Repository --json isDraft 2> $null
if ($LASTEXITCODE -eq 0) {
    $existingRelease = ($existingReleaseJson -join "") | ConvertFrom-Json
    if (-not $existingRelease.isDraft) {
        throw "Published Release already exists: $Repository tag $releaseTag"
    }
}

$notes = @"
Parapperで音声認識をする際、日本語の文末を文法的に解析して区切るために使う辞書です。

[UniDic CWJ 3.1.1](https://clrd.ninjal.ac.jp/unidic_archive/cwj/3.1.1/)を加工し、
Parapperに必要な項目のみに絞って軽量化しています。

ライセンスは修正BSD（BSD 3-Clause）です。
著作者情報は ``AUTHORS``、ライセンス本文は ``BSD``、加工内容は ``NOTICE`` に記載しています。
確認用のSHA-256 checksumsとして ``SHA256SUMS`` と ``release-manifest.json`` を同梱しています。

このReleaseはParapper本体の更新ではありません。
通常はParapperが取得して使用するため、ユーザーが手動でダウンロードまたは展開する必要はありません。
"@

if ($null -eq $existingRelease) {
    gh release create $releaseTag `
        --repo $Repository `
        --target $Target `
        --title "Parapper向け形態素解析辞書（UniDic CWJ 3.1.1）" `
        --notes $notes `
        --draft `
        --latest=false
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub draft Release creation failed."
    }
}

gh release upload $releaseTag `
    (Join-Path $output $nativeAsset) `
    (Join-Path $output "AUTHORS") `
    (Join-Path $output "BSD") `
    (Join-Path $output "NOTICE") `
    (Join-Path $output "release-manifest.json") `
    (Join-Path $output "SHA256SUMS") `
    --repo $Repository `
    --clobber
if ($LASTEXITCODE -ne 0) {
    throw "GitHub draft Release asset upload failed; the draft was left unpublished."
}

$downloadRoot = [System.IO.Path]::GetFullPath(
    (Join-Path ([System.IO.Path]::GetTempPath()) ("parapper-dictionary-release-" + [guid]::NewGuid().ToString("N")))
)
try {
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    gh release download $releaseTag --repo $Repository --dir $downloadRoot
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub draft Release download failed; the draft was left unpublished."
    }
    cargo run --release --locked -p parapper-morph-dictionary --bin release_morph_dictionary -- `
        verify $downloadRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Downloaded GitHub draft assets failed verification; the draft was left unpublished."
    }
    gh release edit $releaseTag --repo $Repository --draft=false --latest=false
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub draft verification passed, but publishing failed; the verified draft remains."
    }
} finally {
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if ($downloadRoot.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $downloadRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
