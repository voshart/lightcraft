param(
    [switch]$Build,
    [switch]$Release,
    [switch]$Demo
)

$ErrorActionPreference = 'Stop'
$checkoutRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $checkoutRoot 'target' }
$profile = if ($Release) { 'release' } else { 'debug' }
$binary = Join-Path $buildRoot "$profile\lightcraft.exe"

if ($Build -or -not (Test-Path -LiteralPath $binary)) {
    $rustCargo = (Get-Command cargo -ErrorAction SilentlyContinue).Source
    if (-not $rustCargo) { $rustCargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe' }
    if (-not (Test-Path -LiteralPath $rustCargo)) { throw 'Install the Rust toolchain before building LightCraft.' }
    $env:PATH = (Split-Path -Parent $rustCargo) + ';' + $env:PATH
    if (-not $env:CRAFT_FONTS_DIR) {
        $fontInput = Join-Path (Split-Path -Parent $checkoutRoot) 'craft-fonts'
        if (Test-Path -LiteralPath $fontInput) { $env:CRAFT_FONTS_DIR = $fontInput }
    }
    $env:CARGO_TARGET_DIR = $buildRoot
    $buildArgs = @('build', '--locked', '-p', 'lightcraft', '-p', 'lightcraft-cli')
    if ($Release) { $buildArgs += '--release' }
    Push-Location -LiteralPath $checkoutRoot
    try {
        & $rustCargo @buildArgs
        if ($LASTEXITCODE -ne 0) { throw "LightCraft build failed: $LASTEXITCODE" }
    } finally { Pop-Location }
}

$appArgs = if ($Demo) { @('--memory') } else { @() }
& $binary @appArgs
exit $LASTEXITCODE
