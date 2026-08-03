# Audited CAT-Translate 0.8B ONNX exporter.
#
# The publish candidate is -Variant k_quant. Other variants are diagnostic and
# deliberately do not receive publication metadata or checksums.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "fp32", "fp16-dml", "q4-default", "q4-accuracy1",
        "q4-exclude-lm-head", "rtn_last", "k_quant", "k_quant_mixed",
        "k_quant_last"
    )]
    [string]$Variant,

    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $true)]
    [string]$CacheDir,

    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SourceRepository = "cyberagent/CAT-Translate-0.8b"
$SourceRevision = "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9"
$RuntimeFiles = @(
    "chat_template.jinja",
    "genai_config.json",
    "model_q4.onnx",
    "model_q4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentScript = Join-Path $ScriptDir "cat_export_environment.py"
$SourceVerifierScript = Join-Path $ScriptDir "verify_cat_source_snapshot.py"
$VerifierScript = Join-Path $ScriptDir "verify_cat_onnx_distribution.py"
$EmbeddingQuantizerScript = Join-Path $ScriptDir "quantize_cat_embedding_gather.py"
$AssetsDir = Join-Path $ScriptDir "assets"

function Resolve-RequiredFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist or is not a file: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-RequiredDirectory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist or is not a directory: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $Encoding)
}

function Test-IsSameOrAncestor([string]$Candidate, [string]$Protected) {
    $Candidate = $Candidate.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $Protected = [System.IO.Path]::GetFullPath($Protected)
    return $Protected.Equals($Candidate, [System.StringComparison]::OrdinalIgnoreCase) -or
        $Protected.StartsWith(
            $Candidate + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )
}

$PythonPath = Resolve-RequiredFile $PythonPath "PythonPath"
$SourceDir = Resolve-RequiredDirectory $SourceDir "SourceDir"
& $PythonPath $SourceVerifierScript $SourceDir
if ($LASTEXITCODE -ne 0) {
    throw "SourceDir does not match the exact audited CAT-Translate source snapshot."
}
$SourceConfig = Get-Content -LiteralPath (Join-Path $SourceDir "config.json") -Raw | ConvertFrom-Json
if (
    $SourceConfig.architectures[0] -ne "LlamaForCausalLM" -or
    $SourceConfig.model_type -ne "llama" -or
    $SourceConfig.hidden_size -ne 1280 -or
    $SourceConfig.num_hidden_layers -ne 24 -or
    $SourceConfig.vocab_size -ne 102400 -or
    $SourceConfig.tie_word_embeddings -ne $false -or
    $SourceConfig.transformers_version -ne "4.57.6"
) {
    throw "source config does not match the audited CAT-Translate 0.8B revision"
}

$EnvironmentJson = (& $PythonPath $EnvironmentScript)
if ($LASTEXITCODE -ne 0) {
    throw "Python export environment does not match requirements-cat-onnx.txt"
}
$Environment = ($EnvironmentJson -join "") | ConvertFrom-Json

$OutDir = [System.IO.Path]::GetFullPath($OutDir)
$CacheDir = [System.IO.Path]::GetFullPath($CacheDir)
$FileSystemRoot = [System.IO.Path]::GetPathRoot($OutDir)
$ProtectedPaths = @(
    $SourceDir,
    $ScriptDir,
    $PythonPath,
    $CacheDir,
    (Get-Location).Path
)
if (
    $OutDir.Equals($FileSystemRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    ($ProtectedPaths | Where-Object { Test-IsSameOrAncestor $OutDir $_ })
) {
    throw "Unsafe OutDir would delete or contain a required input/workspace path: $OutDir"
}
if (
    (Test-IsSameOrAncestor $OutDir $CacheDir) -or
    (Test-IsSameOrAncestor $CacheDir $OutDir)
) {
    throw "CacheDir and OutDir must be disjoint so cache files cannot enter the distribution"
}
if (Test-Path -LiteralPath $OutDir) {
    if (-not $Force) {
        throw "OutDir already exists; choose a clean directory or pass -Force: $OutDir"
    }
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null

$BuilderOutDir = $OutDir
if ($Variant -eq "k_quant") {
    $BuilderOutDir = Join-Path `
        (Split-Path -Parent $OutDir) `
        (".cat-embedding-fp32-" + [Guid]::NewGuid().ToString("N"))
}
New-Item -ItemType Directory -Path $BuilderOutDir -Force | Out-Null

$Precision = "int4"
$ExecutionProvider = "cpu"
$Filename = "model_q4.onnx"
$ExtraOptions = @("filename=$Filename", "hf_token=false", "hf_remote=false")
switch ($Variant) {
    "fp32" {
        $Precision = "fp32"
        $Filename = "model_fp32.onnx"
        $ExtraOptions[0] = "filename=$Filename"
    }
    "fp16-dml" {
        $Precision = "fp16"
        $ExecutionProvider = "dml"
        $Filename = "model_fp16.onnx"
        $ExtraOptions[0] = "filename=$Filename"
    }
    "q4-default" {}
    "q4-accuracy1" { $ExtraOptions += "int4_accuracy_level=1" }
    "q4-exclude-lm-head" { $ExtraOptions += "int4_nodes_to_exclude=/lm_head/MatMul" }
    default { $ExtraOptions += "int4_algo_config=$Variant" }
}

$BuilderArgs = @(
    "-m", "onnxruntime_genai.models.builder",
    "-i", $SourceDir,
    "-o", $BuilderOutDir,
    "-c", $CacheDir,
    "-p", $Precision,
    "-e", $ExecutionProvider,
    "--extra_options"
) + $ExtraOptions

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$SensitivePaths = @($PythonPath, $SourceDir, $OutDir, $BuilderOutDir, $CacheDir)
try {
    & $PythonPath @BuilderArgs 2>&1 | ForEach-Object {
        $Line = $_.ToString()
        foreach ($SensitivePath in $SensitivePaths) {
            $Line = $Line.Replace($SensitivePath, "<LOCAL_PATH>")
        }
        Write-Host $Line
    }
    $BuilderExitCode = $LASTEXITCODE
    if ($BuilderExitCode -ne 0) {
        $Stopwatch.Stop()
        throw "onnxruntime-genai builder failed with exit code $BuilderExitCode"
    }

    if ($Variant -ne "k_quant") {
        $Stopwatch.Stop()
        Write-Host "diagnostic export complete; publication metadata is generated only for -Variant k_quant"
        Write-Host ("duration_seconds={0:N1}" -f $Stopwatch.Elapsed.TotalSeconds)
        exit 0
    }

    $SanitizedCommand = @("python") + @(
        $BuilderArgs | ForEach-Object {
            if ($_ -eq $SourceDir) { "<SOURCE_DIR>" }
            elseif ($_ -eq $BuilderOutDir) { "<INTERMEDIATE_DIR>" }
            elseif ($_ -eq $CacheDir) { "<CACHE_DIR>" }
            else { $_ }
        }
    )

    foreach ($Name in $RuntimeFiles) {
        $Path = Join-Path $BuilderOutDir $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "builder output is incomplete; missing $Name"
        }
    }

    & $PythonPath $EmbeddingQuantizerScript `
        $BuilderOutDir `
        $OutDir `
        --block-size 16
    if ($LASTEXITCODE -ne 0) {
        throw "Q4 block16 embedding quantization failed"
    }
}
finally {
    if (
        $Variant -eq "k_quant" -and
        (Test-Path -LiteralPath $BuilderOutDir)
    ) {
        Remove-Item -LiteralPath $BuilderOutDir -Recurse -Force
    }
}
$Stopwatch.Stop()

Copy-Item -LiteralPath (Join-Path $SourceDir "LICENSE") -Destination (Join-Path $OutDir "LICENSE")
Copy-Item -LiteralPath (Join-Path $AssetsDir "cat-translate-0.8b-q4-k-quant-model-card.md") -Destination (Join-Path $OutDir "MODEL_CARD.md")
Copy-Item -LiteralPath (Join-Path $AssetsDir "cat-translate-0.8b-q4-k-quant-third-party-notices.md") -Destination (Join-Path $OutDir "THIRD_PARTY_NOTICES.md")

$BuildMetadata = [ordered]@{
    schema_version = 1
    source = [ordered]@{
        repository = $SourceRepository
        revision = $SourceRevision
        license = "MIT"
    }
    export = [ordered]@{
        variant = "k_quant"
        precision = "int4"
        execution_provider = "cpu"
        embedding = "groupwise_q4_block16"
        embedding_quantization = [ordered]@{
            bits = 4
            block_size = 16
            is_symmetric = $false
            operator = "GatherBlockQuantized"
            command = @(
                "python",
                "quantize_cat_embedding_gather.py",
                "<INTERMEDIATE_DIR>",
                "<OUT_DIR>",
                "--block-size",
                "16"
            )
        }
        command = $SanitizedCommand
        duration_seconds = [Math]::Round($Stopwatch.Elapsed.TotalSeconds, 3)
    }
    environment = $Environment
}
$MetadataJson = $BuildMetadata | ConvertTo-Json -Depth 8
Write-Utf8NoBom (Join-Path $OutDir "build-metadata.json") ($MetadataJson + "`n")

& $PythonPath $VerifierScript $OutDir --write-manifest
if ($LASTEXITCODE -ne 0) {
    throw "publish candidate verification failed"
}

Write-Host "publish candidate export complete"
Write-Host ("duration_seconds={0:N1}" -f $Stopwatch.Elapsed.TotalSeconds)
