[CmdletBinding()]
param(
    [string]$PortableRoot = 'D:\Comfy\ComfyUI_windows_portable',
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

function Resolve-ComfyCore {
    param([string]$Candidate)

    $resolved = (Resolve-Path -LiteralPath $Candidate).Path
    if (Test-Path -LiteralPath (Join-Path $resolved 'comfy\supported_models.py')) {
        return $resolved
    }

    $nested = Join-Path $resolved 'ComfyUI'
    if (Test-Path -LiteralPath (Join-Path $nested 'comfy\supported_models.py')) {
        return $nested
    }

    throw "ComfyUI core was not found under '$Candidate'."
}

function Test-FeatureMarkers {
    param([string]$Core)

    $detection = Get-Content -LiteralPath (Join-Path $Core 'comfy\model_detection.py') -Raw
    $nodes = Get-Content -LiteralPath (Join-Path $Core 'comfy_extras\nodes_wan.py') -Raw
    return $detection.Contains('dit_config["vace_image_input"] = True') -and
        $nodes.Contains('ref_pad_image')
}

$core = Resolve-ComfyCore -Candidate $PortableRoot
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$patch = Join-Path $kit 'kijai-id-v2v.patch'
$git = Get-Command git.exe -ErrorAction Stop

Write-Host "ComfyUI core: $core"
Write-Host 'Patch source: Kijai ID-V2V commit 7fb4aefac072720a4bbc75e276b8fb71cd031875'

if (Test-FeatureMarkers -Core $core) {
    Write-Host 'ID-V2V feature markers are already present. Nothing to do.' -ForegroundColor Green
    exit 0
}

& $git.Source -C $core apply --3way --check --whitespace=nowarn $patch
if ($LASTEXITCODE -ne 0) {
    throw @"
The exact Kijai diff does not apply cleanly, so nothing was changed.
Update the Portable core with its canonical update\update_comfyui.bat, then run this check again.
"@
}

if (-not $Apply) {
    Write-Host 'Check passed. Re-run with -Apply to create a backup and apply the four-file diff.' -ForegroundColor Cyan
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $core ".kijai-id-v2v-backup\$stamp"
$relativeFiles = @(
    'comfy\ldm\wan\model.py',
    'comfy\model_detection.py',
    'comfy\supported_models.py',
    'comfy_extras\nodes_wan.py'
)

foreach ($relative in $relativeFiles) {
    $source = Join-Path $core $relative
    $destination = Join-Path $backup $relative
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

& $git.Source -C $core apply --3way --whitespace=nowarn $patch
if ($LASTEXITCODE -ne 0) {
    throw "Patch application failed after the backup was written to '$backup'."
}

if (-not (Test-FeatureMarkers -Core $core)) {
    throw "Patch command returned success, but feature-marker verification failed. Backup: '$backup'."
}

Write-Host 'Kijai ID-V2V support was applied successfully.' -ForegroundColor Green
Write-Host "Backup: $backup"
Write-Host 'ComfyUI was not started, no model was loaded, and no custom nodes were installed.'
