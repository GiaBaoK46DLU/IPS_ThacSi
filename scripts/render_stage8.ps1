param(
    [string]$Deck = 'docs/lessons/chang_08/presentation/IPS_KAN_Mixer_UJIIndoorLoc.pptx',
    [string]$PreviewDir = 'runs/stage8_build/slides'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$deckPath = (Resolve-Path (Join-Path $projectRoot $Deck)).Path
$previewPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $PreviewDir))
$pdfPath = [IO.Path]::ChangeExtension($deckPath, '.pdf')
New-Item -ItemType Directory -Force -Path $previewPath | Out-Null
$stage8PowerPoint = $null
$stage8Presentation = $null
try {
    $stage8PowerPoint = New-Object -ComObject PowerPoint.Application
    # Open read-only, without an interactive window. Native Office renders the PPTX.
    $stage8Presentation = $stage8PowerPoint.Presentations.Open($deckPath, -1, 0, 0)
    if ($stage8Presentation.Slides.Count -ne 20) { throw 'Expected 20 slides' }
    $stage8Presentation.SaveAs($pdfPath, 32)
    foreach ($slide in $stage8Presentation.Slides) {
        $pngPath = Join-Path $previewPath ('slide-{0:D2}.png' -f $slide.SlideIndex)
        $slide.Export($pngPath, 'PNG', 1600, 900)
    }
    Write-Output "Rendered $($stage8Presentation.Slides.Count) slides and PDF: $pdfPath"
} finally {
    if ($null -ne $stage8Presentation) { $stage8Presentation.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($stage8Presentation) }
    if ($null -ne $stage8PowerPoint) { $stage8PowerPoint.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($stage8PowerPoint) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
