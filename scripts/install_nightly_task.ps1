# Register the nightly training run as a Windows scheduled task.
#
# WHY A SCHEDULED TASK AND NOT JUST LEAVING A TERMINAL OPEN
# ---------------------------------------------------------
# The machine this trains on gets restarted during the day, is occasionally
# shut down overnight, and rarely goes off for several days. A terminal
# window survives none of that. A scheduled task starts itself at midnight
# every day regardless of what happened before, and because the trainer
# resumes from its last checkpoint, an interrupted night costs only the
# minutes since the last save (~5, at --checkpoint-interval 100).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_nightly_task.ps1
#
# Remove it with:
#   Unregister-ScheduledTask -TaskName Minerva-Nightly-Training -Confirm:$false

param(
    [string]$TaskName = "Minerva-Nightly-Training",
    [string]$StartTime = "00:00",
    [string]$EndTime   = "07:15",
    [int]$Threads      = 10
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$log = Join-Path $repo "data\train_nightly.log"

Write-Host "repo    $repo"
Write-Host "python  $python"
Write-Host "window  $StartTime - $EndTime, $Threads threads"

# -u so the log is written as training runs rather than buffered until exit.
$inner = "& '$python' -u scripts/train_nightly.py --from $StartTime --to $EndTime " +
         "--threads $Threads *>> '$log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$inner`"" `
    -WorkingDirectory $repo

# Two triggers: the nightly start, and one at logon. The logon trigger is what
# makes a daytime restart recover on its own - train_nightly.py sleeps
# immediately if it wakes outside the window, so starting it at noon is
# harmless and it will be running when midnight arrives.
#
# The logon trigger is scoped to the current user on purpose. Without -User it
# means "any user logs on", which is a machine-wide change and needs an
# elevated shell; scoped to one account it registers as a normal user, so this
# script needs no administrator rights.
#
# The third trigger is a self-heal. The run is an ordinary process, so closing
# its console window - or any other accidental kill - ends it silently, and
# nobody would notice until the next morning's status check. Retrying every 30
# minutes across the day means the worst case is half an hour lost rather than
# a whole night. It is safe to fire while training is already running because
# MultipleInstances IgnoreNew (below) refuses to start a second copy, and safe
# to fire outside the window because train_nightly.py sleeps when it wakes
# there.
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
    -Settings $settings -Description "Minerva Swift pretraining, nightly window" -Force | Out-Null

Write-Host ""
Write-Host "Registered '$TaskName'."
Write-Host "  log        $log"
Write-Host "  start now  Start-ScheduledTask -TaskName $TaskName"
Write-Host "  status     Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  remove     Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
