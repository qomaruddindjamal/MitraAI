param (
    [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
    [string[]]$PromptArgs
)

if ($PromptArgs.Count -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "       MITRAAI CLOUD CHAT (QWEN3 RUNNER GITHUB)            " -ForegroundColor Cyan
    Write-Host "       Ketik 'exit' untuk selesai                          " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    while ($true) {
        Write-Host "Anda > " -NoNewline -ForegroundColor Green
        $userInput = Read-Host
        if ([string]::IsNullOrWhiteSpace($userInput)) {
            continue
        }
        if ($userInput.Trim() -eq "exit" -or $userInput.Trim() -eq "quit") {
            Write-Host "Sampai jumpa!" -ForegroundColor Yellow
            break
        }

        Write-Host "Mengirim pertanyaan ke Qwen di Cloud GitHub..." -ForegroundColor DarkGray
        $runOutput = gh workflow run "MitraAI API Runner" -f prompt="$userInput" 2>&1
        $runIdMatch = $runOutput | Select-String -Pattern "/runs/(\d+)"
        if ($runIdMatch) {
            $runId = $runIdMatch.Matches[0].Groups[1].Value
            Write-Host "Menunggu jawaban Qwen (Run #$runId)..." -ForegroundColor DarkGray
            gh run watch $runId --exit-status > $null 2>&1
            Write-Host ""
            Write-Host "MitraAI > " -ForegroundColor Magenta
            gh run view $runId --log | Select-String -Pattern "MitraAI >" -Context 0, 15 | ForEach-Object { $_.Line }
            Write-Host ""
        } else {
            Write-Host "Gagal mengirim workflow. Pastikan koneksi dan gh CLI terhubung." -ForegroundColor Red
        }
    }
} else {
    $fullPrompt = $PromptArgs -join " "
    Write-Host "Mengirim ke Qwen: '$fullPrompt'..." -ForegroundColor DarkGray
    $runOutput = gh workflow run "MitraAI API Runner" -f prompt="$fullPrompt" 2>&1
    $runIdMatch = $runOutput | Select-String -Pattern "/runs/(\d+)"
    if ($runIdMatch) {
        $runId = $runIdMatch.Matches[0].Groups[1].Value
        Write-Host "Menunggu respon Qwen di Cloud..." -ForegroundColor DarkGray
        gh run watch $runId --exit-status > $null 2>&1
        Write-Host ""
        Write-Host "MitraAI > " -ForegroundColor Magenta
        gh run view $runId --log | Select-String -Pattern "MitraAI >" -Context 0, 15 | ForEach-Object { $_.Line }
        Write-Host ""
    }
}
