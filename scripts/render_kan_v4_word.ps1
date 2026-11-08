$ErrorActionPreference = 'Stop'
$kanTaskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$kanDocxPath = Join-Path $kanTaskRoot 'references\KAN-mixer\IPS_KAN_V4.docx'
$kanQaPath = Join-Path $kanTaskRoot 'runs\kan_v4_docx_20261026\render'
New-Item -ItemType Directory -Path $kanQaPath -Force | Out-Null
$kanBeforeWord = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$kanWordApp = $null
$kanWordDoc = $null
$kanOwnWord = $false
try {
    Write-Output 'Starting private Word renderer'
    $kanWordApp = New-Object -ComObject Word.Application
    $kanWordVersion = $kanWordApp.Version
    $kanNewWord = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $kanBeforeWord -notcontains $_.Id })
    if ($kanNewWord.Count -ne 1) { throw 'Cannot establish a new private Word process; refusing to change preexisting Word state.' }
    $kanOwnWord = $true
    $kanWordApp.Visible = $false
    $kanWordApp.DisplayAlerts = 0
    $kanWordApp.AutomationSecurity = 3
    Write-Output 'Opening generated document'
    $kanWordDoc = $kanWordApp.Documents.Open($kanDocxPath,$false,$false,$false)
    Write-Output 'Updating fields and pagination'
    $kanWordDoc.Fields.Update() | Out-Null
    $kanWordDoc.Repaginate()
    foreach ($kanToc in $kanWordDoc.TablesOfContents) { $kanToc.Update() }
    $kanWordDoc.Repaginate()
    foreach ($kanToc in $kanWordDoc.TablesOfContents) { $kanToc.UpdatePageNumbers() }
    $kanWordDoc.Save()
    Write-Output 'DOCX saved with updated contents'
    $kanWordDoc.Close(0)
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($kanWordDoc) | Out-Null
    $kanWordDoc = $null
    $kanWordDoc = $kanWordApp.Documents.Open($kanDocxPath,$false,$true,$false)
    $kanPdfPath = Join-Path $kanQaPath 'IPS_KAN_V4.pdf'
    $kanWordDoc.ExportAsFixedFormat($kanPdfPath,17)
    Write-Output 'PDF exported'
    $kanHeadingRows = @()
    foreach ($kanParagraph in $kanWordDoc.Paragraphs) {
        if ($kanParagraph.OutlineLevel -eq 1) {
            $kanHeadingRows += [pscustomobject]@{ text = $kanParagraph.Range.Text.Trim(); page = $kanParagraph.Range.Information(3) }
        }
    }
    $kanLayout = [pscustomobject]@{
        renderer = 'Microsoft Word private invisible COM process'
        reason = 'Packaged render_docx.py reports missing LibreOffice; Windows runtime does not bundle LibreOffice.'
        wordVersion = $kanWordApp.Version
        pages = $kanWordDoc.ComputeStatistics(2)
        words = $kanWordDoc.ComputeStatistics(0)
        nativeEquations = $kanWordDoc.OMaths.Count
        tables = $kanWordDoc.Tables.Count
        toc = $kanWordDoc.TablesOfContents.Count
        headings = $kanHeadingRows
        pdf = $kanPdfPath
    }
    $kanLayout | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $kanQaPath 'layout.json') -Encoding UTF8
    $kanLayout | Select-Object pages,words,nativeEquations,tables,toc,pdf | ConvertTo-Json
}
finally {
    if ($null -ne $kanWordDoc) { $kanWordDoc.Close(0); [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($kanWordDoc) | Out-Null }
    if ($null -ne $kanWordApp) {
        if ($kanOwnWord) { $kanWordApp.Quit(0) }
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($kanWordApp) | Out-Null
    }
}
