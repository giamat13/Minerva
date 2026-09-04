# Register background training as a Windows scheduled task.
#
# WHY A SCHEDULED TASK AND NOT JUST LEAVING A TERMINAL OPEN
# ---------------------------------------------------------
# The machine this trains on gets restarted during the day, is occasionally
# shut down overnight, and rarely goes off for several days. A terminal
# window survives none of that. A scheduled task starts itself regardless of
# what happened before, and because the trainer resumes from its last
# checkpoint, an interruption costs only the minutes since the last save
# (~5, at --checkpoint-interval 100).
#
# The task runs train_when_away.py, which trains only while nobody is at the
# computer - asked of Screen Time's local API - and falls back to the
# StartTime-EndTime window when that API cannot be reached.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_training_task.ps1
#
# Remove it with:
#   Unregister-ScheduledTask -TaskName Minerva-Background-Training -Confirm:$false

param(
    [string]$TaskName = "Minerva-Background-Training",
    [string]$StartTime = "00:00",
    [string]$EndTime   = "07:15",
    [int]$Threads      = 10
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$log = Join-Path $repo "data\training.log"

Write-Host "repo    $repo"
Write-Host "python  $python"
Write-Host "presence-driven; fallback window $StartTime - $EndTime, $Threads threads"

# The previous name, from when this was a fixed night window. Removed so the
# two do not both train into the same checkpoint directory.
$old = Get-ScheduledTask -TaskName "Minerva-Nightly-Training" -ErrorAction SilentlyContinue
if ($old) {
    Unregister-ScheduledTask -TaskName "Minerva-Nightly-Training" -Confirm:$false
    Write-Host "removed the old Minerva-Nightly-Training task"
}

# -u so the log is written as training runs rather than buffered until exit.
# Piped through Out-File rather than `*>>`, which writes UTF-16 and made the
# log render as "[ 2 0 2 6 - . . ." in every ordinary text tool.
$inner = "& '$python' -u scripts/train_when_away.py --from $StartTime --to $EndTime " +
         "--threads $Threads *>&1 | Out-File -FilePath '$log' -Encoding utf8 -Append"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$inner`"" `
    -WorkingDirectory $repo

# Three triggers, all of them ways of saying "make sure it is running". The
# script itself decides whether to train or wait, so starting it at any moment
# is harmless: if someone is at the computer it simply watches and waits.
#
# The logon trigger is scoped to the current user on purpose. Without -User it
# means "any user logs on", which is a machine-wide change and needs an
# elevated shell; scoped to one account it registers as a normal user, so this
# script needs no administrator rights.
#
# The repeating trigger is a self-heal. The run is an ordinary process, so
# closing its console window - or any other accidental kill - ends it
# silently, and nobody would notice until the next status check. Retrying
# every 30 minutes means the worst case is half an hour idle rather than days.
# It is safe to fire while training is already running because
# MultipleInstances IgnoreNew (below) refuses to start a second copy.
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration ([TimeSpan]::FromDays(3650))

$triggers = @(
    (New-ScheduledTaskTrigger -Daily -At $StartTime),
    (New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"),
    $repeating
)

# StopIfGoingOnBatteries stays off on purpose: on a desktop it is irrelevant,
# and on a laptop the user asked for the run to continue. ExecutionTimeLimit 0
# means "no limit" - the script manages its own window and would otherwise be
# killed at Windows' 72-hour default mid-checkpoint.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "Minerva Swift pretraining, runs while nobody is at the computer" -Force | Out-Null

Write-Host ""
Write-Host "Registered '$TaskName'."
Write-Host "  log        $log"
Write-Host "  start now  Start-ScheduledTask -TaskName $TaskName"
Write-Host "  status     Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  remove     Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
