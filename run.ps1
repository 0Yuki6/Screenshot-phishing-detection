# PowerShell runner for Phishing Screenshot Detector tasks
param (
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Command = "test"
)

python tasks.py $Command
