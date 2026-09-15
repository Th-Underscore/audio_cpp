#!/usr/local/bin/pwsh -File

[CmdletBinding()]
param()

cd -lit $PSScriptRoot
$file = "audio_cpp-v0.1.0.7z"

$dirs += (gci | ? { $_ -match ".*\.py$|.*\.js$" } | % { $_.Name })
$dirs += @("requirements.txt", "DESIGN.md", "model_specs/")
#$dirs += @("docs/")

echo $dirs

Write-Verbose "1"

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Verbose "2"

function ConvertTo-IgnoreRegex {
    param([string]$Pattern)
    $anchored = $Pattern.Contains('/')
    $p = $Pattern.TrimEnd('/').TrimStart('/')
    $r = [regex]::Escape($p) `
        -replace '\\\*\\\*/', '(?:.+/)?' `
        -replace '\\\*\\\*', '.*' `
        -replace '\\\*', '[^/]*' `
        -replace '\\\?', '[^/]'
    if ($anchored) { "^$r(/.*)?$" } else { "^(?:.+/)?$r(/.*)?$" }
}

function Test-Ignored {
    param([string]$RelPath, [string[]]$Rules)
    $ignored = $false
    foreach ($rule in $Rules) {
        if ($rule -match '^\!(.+)') {
            if ($RelPath -match (ConvertTo-IgnoreRegex $Matches[1])) { $ignored = $false }
        } else {
            if ($RelPath -match (ConvertTo-IgnoreRegex $rule)) { $ignored = $true  }
        }
    }
    return $ignored
}

Write-Verbose "3"

$ignoreRules = @()
if (Test-Path '.ignore') {
    $ignoreRules = Get-Content '.ignore' | Where-Object { $_ -notmatch '^\s*(#|$)' }
}
Write-Verbose "4"

try {
    Write-Verbose "$dirs"
    $dirs | ForEach-Object {
        $p = $_.TrimEnd('/')
        Write-Verbose $p
        if (Test-Path -lit $p -PathType Container) { Get-ChildItem -Recurse -File $p }
        elseif (Test-Path -lit $p -PathType Leaf) { Get-Item $p }
    } | ForEach-Object {
        $rel = (Resolve-Path $_.FullName -Relative).Replace('\', '/').TrimStart('./')
        if (-not (Test-Ignored $rel $ignoreRules)) { $rel }
    } | Out-File -lit backup_files.txt -Encoding UTF8
    Write-Verbose "5"

	#7z a -tzip -mx=9 -mm=Deflate -mpass=15 dayna_ss.zip "@backup_files.txt"
    ppmd "$file" "@backup_files.txt"
    Write-Verbose "6"
} finally {
    $BackupExitCode = $LASTEXITCODE
    $ErrorActionPreference = "SilentlyContinue"
    Remove-Item -lit backup_files.txt
    $stopwatch.Stop()
    Write-Host "Execution time: $($stopwatch.Elapsed.TotalSeconds) seconds"
    if ($BackupExitCode -ne 0) {
        #Remove-Item -lit $file
        Write-Error "Backup failed."
        Write-Host "Press any key to continue..."
        [void][System.Console]::ReadKey($true)
    }
}


