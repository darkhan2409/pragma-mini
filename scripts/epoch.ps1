<#
.SYNOPSIS
    Запуск и обслуживание длинного обучения mini_pragma_v2.

.DESCRIPTION
    Обучение идёт отдельным локальным процессом. Ему не нужен
    ни интернет, ни открытая сессия Claude, ни это окно: после
    start сюда можно не возвращаться, а состояние читать
    командой status.

    Два каталога, и это важно:

        run\        каталог модели. Его заводит сам trainer и
                    отказывается работать, если он не пуст.
                    Launcher в него НИЧЕГО не пишет.

        control\    файлы запуска: перенаправленный вывод и PID
                    процесса. Их пишет launcher до старта
                    trainer, поэтому они обязаны лежать в
                    стороне.

    Что где:

        control\stdout.log   что печатало обучение
        control\stderr.log   ошибки и traceback
        control\pid.txt      номер процесса, его пишет launcher
        run\log.jsonl        события: старт, шаги, оценки, checkpoint
        run\status.json      где процесс сейчас, обновляется на ходу
        run\stop.request     просьба остановиться, её ждёт trainer
        run\last.pt          последнее состояние
        run\best.pt          лучшее по val_time.recent

.EXAMPLE
    .\scripts\epoch.ps1 start
    .\scripts\epoch.ps1 status
    .\scripts\epoch.ps1 logs -Lines 40
    .\scripts\epoch.ps1 stop
    .\scripts\epoch.ps1 resume
    .\scripts\epoch.ps1 resume-interrupted
    .\scripts\epoch.ps1 check

.EXAMPLE
    Проверка самого launcher на крошечном наборе, без эпохи:

    .\scripts\epoch.ps1 start -Preset smoke
    .\scripts\epoch.ps1 status -Preset smoke
    .\scripts\epoch.ps1 stop -Preset smoke
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'status', 'logs', 'stop', 'resume', 'resume-interrupted', 'check')]
    [string] $Action = 'status',

    [ValidateSet('epoch', 'smoke')]
    [string] $Preset = 'epoch',

    [int] $Lines = 20,

    [int] $StopTimeoutSeconds = 900
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root 'venv\Scripts\python.exe'

# ============================================================
# КОНФИГ
# ============================================================
#
# Один список аргументов на все команды: start и resume обязаны
# запускать ОДНО И ТО ЖЕ обучение, иначе продолжение отвергнет
# проверка совместимости.
#
# smoke это не эксперимент, а проверка самого launcher: те же
# режимы, но крошечный набор и CPU.

if ($Preset -eq 'epoch') {

    $Name = 'v21_10k'

    $TrainArgs = @(
        '-u', '-m', 'src.model.train', 'run',
        '--name', $Name,
        '--device', 'cuda',
        '--structure', 'session',
        '--d-model', '128',
        '--n-heads', '4',
        '--dim-feedforward', '512',
        '--profile-layers', '1',
        '--event-layers', '3',
        '--session-layers', '1',
        '--history-layers', '2',
        '--max-events', 'none',
        '--train-clients', 'none',
        '--val-clients', 'none',
        '--masking-mode', 'combined',
        '--token-rate', '0.15',
        '--event-rate', '0.10',
        '--key-rate', '0.10',
        '--mask-scheme', 'example',
        '--target-policy', 'history',
        '--epochs', '1',
        '--stream-validation',
        '--best-metric', 'recent',
        '--final-splits', 'test_client,test_time',
        '--batch-size', '2',
        '--eval-batch-size', '2',
        '--eval-every', '20000',
        '--log-every', '200',
        '--checkpoint-every', '2000'
    )
}
else {

    $Name = 'smoke'

    $TrainArgs = @(
        '-u', '-m', 'src.model.train', 'run',
        '--name', $Name,
        '--device', 'cpu',
        '--precision', 'float32',
        '--train-clients', '6',
        '--val-clients', '2',
        '--max-events', '32',
        '--masking-mode', 'combined',
        '--mask-scheme', 'example',
        '--target-policy', 'history',
        '--stream-validation',
        '--best-metric', 'recent',
        '--batch-size', '2',
        '--eval-batch-size', '2',
        '--max-steps', '3000',
        '--warmup-steps', '10',
        '--eval-every', '100000',
        '--log-every', '50',
        '--checkpoint-every', '100'
    )
}

$RunDir = Join-Path $Root "data\runs\$Name\run"
$ControlDir = Join-Path $Root "data\runs\$Name\control"

# Launcher пишет только сюда.
$StdOut = Join-Path $ControlDir 'stdout.log'
$StdErr = Join-Path $ControlDir 'stderr.log'
$PidFile = Join-Path $ControlDir 'pid.txt'

# Это пишет и читает сам trainer.
$StatusFile = Join-Path $RunDir 'status.json'
$StopFile = Join-Path $RunDir 'stop.request'
$LastCheckpoint = Join-Path $RunDir 'last.pt'
$Interrupted = Join-Path $RunDir 'interrupted.pt'

# ============================================================
# ПРОЦЕСС
# ============================================================

function Get-TrainingProcess {
    <#
        Живой процесс этого запуска или $null.

        PID из файла мало: номер переиспользуется системой.
        Поэтому дополнительно проверяется, что это python.
    #>

    if (-not (Test-Path $PidFile)) { return $null }

    $recorded = (Get-Content $PidFile -Raw).Trim()

    if (-not $recorded) { return $null }

    $process = Get-Process -Id ([int] $recorded) -ErrorAction SilentlyContinue

    if ($null -eq $process) { return $null }

    if ($process.ProcessName -notlike 'python*') { return $null }

    return $process
}

function Start-Training {
    param([string[]] $Extra)

    $running = Get-TrainingProcess

    if ($null -ne $running) {
        Write-Host "уже идёт: PID $($running.Id), запущен $($running.StartTime)"
        Write-Host "остановить: .\scripts\epoch.ps1 stop -Preset $Preset"
        return
    }

    if (-not (Test-Path $Python)) {
        throw "нет интерпретатора $Python"
    }

    $resuming = $Extra.Count -gt 0

    # Каталог модели заводит trainer, и он обязан быть пустым.
    # Launcher его не создаёт и не трогает: иначе собственный
    # stdout.log сделал бы запуск невозможным.
    if (-not $resuming -and (Test-Path $RunDir)) {

        $existing = Get-ChildItem $RunDir -Force

        if ($existing.Count -gt 0) {
            Write-Host "каталог $RunDir не пуст, trainer его не перезапишет:"
            $existing | ForEach-Object { Write-Host "  $($_.Name)" }
            Write-Host ''
            Write-Host 'это результат прошлого запуска. Уберите его в сторону вручную'
            Write-Host 'или продолжите: .\scripts\epoch.ps1 resume'
            return
        }
    }

    New-Item -ItemType Directory -Force -Path $ControlDir | Out-Null

    if (Test-Path $StopFile) { Remove-Item $StopFile -Force }

    $arguments = $TrainArgs + $Extra

    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    # Логи дописываются: продолжение не должно стирать историю
    # прошлой попытки.
    Add-Content -Path $StdOut -Value "===== $stamp $($arguments -join ' ') =====" -Encoding utf8

    if (-not (Test-Path $StdErr)) { New-Item -ItemType File -Path $StdErr | Out-Null }

    $started = @{
        FilePath               = $Python
        ArgumentList           = $arguments
        WorkingDirectory       = $Root
        RedirectStandardOutput = $StdOut
        RedirectStandardError  = $StdErr
        NoNewWindow            = $true
        PassThru               = $true
    }

    $process = Start-Process @started

    Set-Content -Path $PidFile -Value $process.Id -Encoding ascii

    Write-Host "запущено: PID $($process.Id)"
    Write-Host "модель:   $RunDir"
    Write-Host "логи:     $ControlDir"
    Write-Host "статус:   .\scripts\epoch.ps1 status -Preset $Preset"
}

# ============================================================
# КОМАНДЫ
# ============================================================

switch ($Action) {

    'start' {
        Start-Training -Extra @()
    }

    'resume' {
        if (-not (Test-Path $LastCheckpoint)) { throw "нет $LastCheckpoint" }
        Start-Training -Extra @('--resume', $LastCheckpoint)
    }

    'resume-interrupted' {
        if (-not (Test-Path $Interrupted)) { throw "нет $Interrupted" }
        Start-Training -Extra @('--resume', $Interrupted)
    }

    'status' {

        $process = Get-TrainingProcess

        if ($null -eq $process) {
            Write-Host 'процесс: не запущен'
        }
        else {
            $minutes = ((Get-Date) - $process.StartTime).TotalMinutes
            $memory = $process.WorkingSet64 / 1MB
            Write-Host ("процесс: PID {0}, работает {1:N0} мин, память {2:N0} МБ" -f $process.Id, $minutes, $memory)
        }

        if (Test-Path $StatusFile) {

            $status = Get-Content $StatusFile -Raw | ConvertFrom-Json

            Write-Host ''
            Write-Host "фаза:             $($status.phase)"
            Write-Host "шаг:              $($status.step) из $($status.steps_per_epoch)"

            if ($null -ne $status.epoch_share) {
                Write-Host ("эпоха:            {0:P1}" -f $status.epoch_share)
            }

            Write-Host ("прошло:           {0:N0} мин" -f ($status.elapsed_seconds / 60))

            if ($null -ne $status.eta_seconds) {
                Write-Host ("осталось:         {0:N0} мин" -f ($status.eta_seconds / 60))
            }

            Write-Host "последняя оценка: $($status.last_validation_step)"
            Write-Host "следующая:        $($status.next_validation_step)"
            Write-Host "checkpoint:       шаг $($status.last_checkpoint.step)"
            Write-Host "лучший:           шаг $($status.best.step), значение $($status.best.value)"

            if ($status.resumed_from) {
                Write-Host "продолжено с:     $($status.resumed_from)"
            }
        }
        else {
            Write-Host "нет $StatusFile (обучение ещё не дошло до первой записи)"
        }
    }

    'logs' {

        foreach ($file in @($StdOut, $StdErr)) {
            Write-Host ''
            Write-Host "--- $file"
            if (Test-Path $file) {
                Get-Content $file -Tail $Lines
            }
            else {
                Write-Host '(нет)'
            }
        }
    }

    'stop' {

        $process = Get-TrainingProcess

        if ($null -eq $process) {
            Write-Host 'процесс не запущен'
            break
        }

        # Мягкая остановка: файл-просьба в каталоге модели,
        # там его и ждёт trainer. Обучение досчитает текущий
        # шаг, сохранит interrupted.pt и выйдет само.
        # Ctrl+C фоновому процессу Windows не доставляет, а
        # Stop-Process убил бы его до сохранения.
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        New-Item -ItemType File -Path $StopFile -Force | Out-Null

        Write-Host "просьба остановиться отправлена: $StopFile"
        Write-Host "ожидание до $StopTimeoutSeconds с (идущая validation досчитается)"

        $deadline = (Get-Date).AddSeconds($StopTimeoutSeconds)

        while ((Get-Date) -lt $deadline) {

            Start-Sleep -Seconds 3

            if ($null -eq (Get-TrainingProcess)) {
                Write-Host 'остановлено, состояние в interrupted.pt'
                Write-Host "продолжить: .\scripts\epoch.ps1 resume-interrupted -Preset $Preset"
                break
            }
        }

        if ($null -ne (Get-TrainingProcess)) {
            Write-Host 'ещё работает: скорее всего идёт validation.'
            Write-Host 'подождите или повторите status; убивать процесс не нужно.'
        }
    }

    'check' {

        $target = if (Test-Path $LastCheckpoint) { $LastCheckpoint }
                  elseif (Test-Path $Interrupted) { $Interrupted }
                  else { $null }

        if ($null -eq $target) {
            Write-Host "в $RunDir нет ни last.pt, ни interrupted.pt"
            break
        }

        & $Python '-u' '-m' 'src.model.train' 'check' '--checkpoint' $target
    }
}
