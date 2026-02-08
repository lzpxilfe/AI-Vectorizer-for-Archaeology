@echo off
set DEST=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer

echo ===================================================
echo  ArchaeoTrace Plugin Deployer
echo ===================================================
echo.
echo Source: %~dp0ai_vectorizer
echo Destination: %DEST%
echo.

if not exist "%DEST%" (
    echo [WARNING] Destination does not exist. Creating it...
    mkdir "%DEST%"
)

echo Copying files...
xcopy "%~dp0ai_vectorizer" "%DEST%" /E /I /Y /Q

echo.
echo ===================================================
echo  Deployment Complete!
echo  Please restart QGIS to apply changes.
echo ===================================================
pause
