# Trading Agent Development Startup Script
# Run this to start both backend and frontend servers

Write-Host "🚀 Starting Trading Agent Development Servers..." -ForegroundColor Cyan
Write-Host ""

# Check if backend .env exists
if (-not (Test-Path "backend\.env")) {
    Write-Host "⚠️  Backend .env not found. Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item "backend\.env.example" "backend\.env"
    Write-Host "✅ Created backend\.env - Please edit it and add your GEMINI_API_KEY" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press any key to continue after editing .env file..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Check if frontend node_modules exists
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location frontend
    npm install
    Set-Location ..
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
    Write-Host ""
}

# Start backend in new window
Write-Host "🔧 Starting Backend Server (Port 8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; Write-Host '🔧 Backend Server Starting...' -ForegroundColor Cyan; python -m app.main"

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start frontend in new window
Write-Host "📱 Starting Frontend Server (Port 5173)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; Write-Host '📱 Frontend Server Starting...' -ForegroundColor Cyan; npm run dev"

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "📊 API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "📱 Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in each terminal window to stop the servers" -ForegroundColor Yellow
Write-Host ""
