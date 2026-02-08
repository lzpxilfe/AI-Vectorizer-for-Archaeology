@echo off
set DEST=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer
set SRC=%~dp0ai_vectorizer

echo ===================================================
echo  ArchaeoTrace Dev Link Setup
echo ===================================================
echo.
echo Removes existing plugin folder and links it to source.
echo.
echo Source: %SRC%
echo Destination: %DEST%
echo.

if exist "%DEST%" (
    echo [INFO] Removing existing folder/link...
    rmdir /S /Q "%DEST%"
)

echo [INFO] Creating Symbolic Link (Junction)...
mklink /J "%DEST%" "%SRC%"

echo.
if errorlevel 0 (
    echo [SUCCESS] Link created! 
    echo Now QGIS will read directly from your source folder.
    echo No need to deploy anymore. Just reload the plugin.
) else (
    echo [ERROR] Failed to create link. Run as Administrator?
)
echo.
pause
