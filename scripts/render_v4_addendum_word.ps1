param([string]$TaskDirectory = 'runs/kan_v4_v2_addendum_20261027')
$ErrorActionPreference = 'Stop'
$v4Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$v4Work = [System.IO.Path]::GetFullPath((Join-Path $v4Root $TaskDirectory))
$v4Input = Join-Path $v4Work 'IPS_KAN_V4.docx'
$v4Pdf = Join-Path $v4Work 'IPS_KAN_V4.pdf'
$v4Before = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$v4Word = $null
$v4Document = $null
$v4OwnWord = $false
try {
    $v4Word = New-Object -ComObject Word.Application
    $v4New = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $v4Before -notcontains $_.Id })
    if ($v4New.Count -ne 1) { throw 'Cannot identify a private Word renderer; preexisting Word state is preserved.' }
    $v4OwnWord = $true
    $v4New[0].Id | Set-Content -LiteralPath (Join-Path $v4Work 'word_renderer.pid')
    $v4Word.Visible = $false
    $v4Word.DisplayAlerts = 0
    $v4Word.AutomationSecurity = 3
    Write-Output 'Opening report read only in a private Word process'
    $v4Document = $v4Word.Documents.Open($v4Input, $false, $true, $false)
    Write-Output 'Updating fields in the read only document'
    $v4Document.Fields.Update() | Out-Null
    $v4Document.Repaginate()
    Write-Output 'Exporting PDF'
    $v4Document.ExportAsFixedFormat($v4Pdf, 17)
    Write-Output 'PDF export complete'
    $v4Headings = @()
    foreach ($v4Paragraph in $v4Document.Paragraphs) {
        if ($v4Paragraph.OutlineLevel -eq 1 -or $v4Paragraph.Range.Text.StartsWith('31.')) {
            $v4Headings += [pscustomobject]@{ text=$v4Paragraph.Range.Text.Trim(); page=$v4Paragraph.Range.Information(3) }
        }
    }
    $v4Layout = [pscustomobject]@{ renderer='Microsoft Word private read only COM';
        packaged_renderer_failure='LibreOffice soffice.exe not installed';
        wordVersion=$v4Word.Version; pages=$v4Document.ComputeStatistics(2);
        words=$v4Document.ComputeStatistics(0); nativeEquations=$v4Document.OMaths.Count;
        tables=$v4Document.Tables.Count; toc=$v4Document.TablesOfContents.Count; headings=$v4Headings }
    $v4Layout | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $v4Work 'layout.json') -Encoding UTF8
    $v4Layout | Select-Object pages,words,nativeEquations,tables,toc | ConvertTo-Json
}
finally {
    if ($null -ne $v4Document) { $v4Document.Close(0); [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($v4Document) | Out-Null }
    if ($null -ne $v4Word) {
        if ($v4OwnWord) { $v4Word.Quit(0) }
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($v4Word) | Out-Null
    }
}
