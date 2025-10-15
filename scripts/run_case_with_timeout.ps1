# scripts/run_case_with_timeout.ps1
# PowerShell wrapper for eng/run_with_timeout.py
# Usage: .\scripts\run_case_with_timeout.ps1 -Timeout 5.0 -Payload '{"method":"adaptive",...}'

param(
    [Parameter(Mandatory=$true)]
    [string]$Payload,
    
    [Parameter(Mandatory=$false)]
    [double]$Timeout = 5.0
)

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$TimeoutRunner = Join-Path $ProjectRoot "eng\run_with_timeout.py"

# Check if the timeout runner exists
if (-not (Test-Path $TimeoutRunner)) {
    Write-Error "Timeout runner not found at: $TimeoutRunner"
    exit 1
}

# Run the timeout wrapper
try {
    $Result = & python $TimeoutRunner --timeout $Timeout --payload $Payload
    $ExitCode = $LASTEXITCODE
    
    # Output the result
    Write-Output $Result
    
    # Exit with the same code as the timeout runner
    exit $ExitCode
}
catch {
    Write-Error "Failed to run timeout wrapper: $_"
    exit 1
}
