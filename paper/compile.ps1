# Compile the LaTeX paper
# Usage: powershell -ExecutionPolicy Bypass -File paper/compile.ps1

$paperDir = "C:\Users\dgonz\PycharmProjects\Tesis_V3\paper"
$pdflatex = "C:\Users\dgonz\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
$texFile  = "paper_triple_screen_ml.tex"
$pdfFile  = "paper_triple_screen_ml.pdf"

$env:TEXINPUTS = "$paperDir//;"
$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== Pass 1/2 ===" -ForegroundColor Cyan
& $pdflatex -interaction=nonstopmode -output-directory="$paperDir" "$paperDir\$texFile" 2>$null | Where-Object { $_ -match "Output written|^!" }

Write-Host "=== Pass 2/2 ===" -ForegroundColor Cyan
& $pdflatex -interaction=nonstopmode -output-directory="$paperDir" "$paperDir\$texFile" 2>$null | Where-Object { $_ -match "Output written|^!" }

$pdf = Join-Path $paperDir $pdfFile
if (Test-Path $pdf) {
    $size = (Get-Item $pdf).Length
    $kb = [math]::Round($size / 1024)
    $logContent = Get-Content "$paperDir\paper_triple_screen_ml.log" -Raw
    if ($logContent -match "Output written.*\((\d+) pages") { $pages = $Matches[1] } else { $pages = "?" }
    Write-Host "=== OK: $pdfFile ($pages pages, ${kb}KB) ===" -ForegroundColor Green
} else {
    Write-Host "=== FAILED: PDF not generated ===" -ForegroundColor Red
    exit 1
}
