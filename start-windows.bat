@echo off
rem snowcoach 一鍵安裝與啟動（Windows）
rem 雙擊執行：第一次會自動安裝（約 5～10 分鐘，需要網路），之後雙擊就直接開啟圖形介面。
rem 所有東西都裝在這個資料夾裡（.venv、models），刪掉資料夾就等於解除安裝。
rem 加上 --check 參數時只安裝並檢查，不開啟介面（給自動測試用）。
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "SNOWCOACH_MODELS=%~dp0models"
set "GLOG_minloglevel=2"
set "PYTHONUTF8=1"
set "CHECK=0"
if /i "%~1"=="--check" set "CHECK=1"

echo ==============================================
echo   snowcoach 單板滑雪動作分析
echo   自動分析，非專業教練建議，僅供參考
echo ==============================================

rem 1. uv：負責下載 Python 與套件的小工具（裝在使用者自己的資料夾，不需要系統管理員權限）
set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if defined UV goto have_uv
echo 第一次使用：安裝必要工具 uv ……
powershell -NoProfile -ExecutionPolicy ByPass -Command "$env:UV_NO_MODIFY_PATH=1; irm https://astral.sh/uv/install.ps1 | iex"
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" goto fail_uv
:have_uv

rem 2. Python 與套件：pyproject.toml 有變（例如更新版本）時重新安裝
for /f "delims=" %%h in ('powershell -NoProfile -Command "(Get-FileHash pyproject.toml).Hash"') do set "HASH=%%h"
set "STAMP=.venv\.snowcoach-installed"
set "OLD="
if exist "%STAMP%" set /p OLD=<"%STAMP%"
if "%OLD%"=="%HASH%" goto installed

echo 安裝 Python 與分析套件，第一次約 5～10 分鐘，請耐心等候……
if exist .venv rmdir /s /q .venv
"%UV%" venv --python 3.11 .venv
if errorlevel 1 goto fail_venv
"%UV%" pip install --python .venv\Scripts\python.exe -e ".[gui,h264]"
if errorlevel 1 goto fail_pip
echo 下載姿勢模型……
.venv\Scripts\snowcoach.exe download-models
if errorlevel 1 goto fail_models
>"%STAMP%" echo %HASH%
echo 安裝完成！之後雙擊就會直接開啟。

:installed
if "%CHECK%"=="0" goto launch
.venv\Scripts\snowcoach.exe --help >nul
if errorlevel 1 goto fail_check
echo 安裝檢查完成
exit /b 0

:launch
echo.
echo 啟動中……瀏覽器會自動打開分析頁面。
echo 分析結果也會存在：%APP_DIR%out\gui
echo 使用完畢後，關閉這個視窗即可結束。
echo.
.venv\Scripts\snowcoach.exe gui
if errorlevel 1 goto fail_run
exit /b 0

:fail_uv
set "MSG=安裝 uv 失敗，請確認網路連線。"
goto fail
:fail_venv
set "MSG=建立 Python 環境失敗。"
goto fail
:fail_pip
set "MSG=安裝套件失敗，請確認網路連線後再試一次。"
goto fail
:fail_models
set "MSG=下載模型失敗，請確認網路連線後再試一次。"
goto fail
:fail_check
set "MSG=安裝檢查失敗。"
goto fail
:fail_run
set "MSG=程式發生錯誤。"
goto fail
:fail
echo.
echo [錯誤] %MSG%
echo 請把這個視窗的內容截圖或複製下來，詢問提供工具的人。
if "%CHECK%"=="0" pause
exit /b 1
