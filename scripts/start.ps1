# Start the complete local UI, authentication and monitoring stack.
# -Live enables real travel providers; -Groq enables hosted narration.
# -Check prepares local files and validates Compose without starting containers.
[CmdletBinding()]
param(
    [switch]$Live,
    [switch]$Groq,
    [switch]$Check,
    [ValidateRange(1, 65535)]
    [int]$GatewayPort = 8000
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$launchEnvironment = @{
    ENVIRONMENT = 'local'
    LLM_PROVIDER = 'mock'
    PROVIDER_MODE = 'mock'
    OTEL_ENABLED = 'true'
    GROQ_API_KEY_FILE_HOST = './.tmp/groq-placeholder.key'
    GATEWAY_HOST_PORT = "$GatewayPort"
}
$previousEnvironment = @{}

Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install Docker Desktop, start it, then run this command again.'
    }
    if ($Groq) {
        if (-not (Test-Path -LiteralPath 'apikeys/api.key' -PathType Leaf)) {
            throw 'Save your Groq key in apikeys/api.key before using -Groq.'
        }
        $launchEnvironment.LLM_PROVIDER = 'groq'
        $launchEnvironment.GROQ_API_KEY_FILE_HOST = './apikeys/api.key'
    }

    New-Item -ItemType Directory -Force '.tmp' | Out-Null
    # These are development placeholders, not provider credentials. Never replace existing files.
    $localFiles = @{
        '.tmp/groq-placeholder.key' = 'unused-in-mock-mode'
        '.tmp/orchestrator-client.key' = 'CHANGE_ME_ORCHESTRATOR_SECRET'
        '.tmp/quickstart.env' = '# Local launch settings are supplied by scripts/start.ps1.'
    }
    foreach ($file in $localFiles.Keys) {
        if (-not (Test-Path -LiteralPath $file)) {
            Set-Content -LiteralPath $file -Value $localFiles[$file] -Encoding ascii
        }
    }

    $composeArguments = @(
        '--env-file', '.tmp/quickstart.env',
        '-f', 'docker-compose.yml', '-f', 'docker-compose.local.yml'
    )
    if ($Live) {
        New-Item -ItemType Directory -Force 'apikeys/duffel', 'apikeys/liteapi', 'apikeys/geoapify' | Out-Null
        $composeArguments += @('-f', 'docker-compose.live.yml')
    }
    $composeArguments += @('--profile', 'full')

    foreach ($name in $launchEnvironment.Keys) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $launchEnvironment[$name], 'Process')
    }
    & docker compose @composeArguments config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose configuration failed validation.' }
    if ($Check) {
        Write-Host 'Configuration valid. No containers were started or changed.'
        return
    }

    Write-Host 'Starting VoyageMesh. The first build can take several minutes.'
    & docker compose @composeArguments up -d --build --wait --wait-timeout 180
    if ($LASTEXITCODE -ne 0) {
        throw 'Startup failed. Check Docker Desktop and the error above, then rerun this command.'
    }
    Write-Host 'Open http://localhost:3000 and sign in with traveller / traveller.'
    if ($Live) {
        Write-Host 'Live mode: providers without keys will report unavailable data.'
    } else {
        Write-Host 'Demo mode: travel prices are simulated.'
    }
} finally {
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
    Pop-Location
}
