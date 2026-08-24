@echo off
setlocal

if defined ARCHAEOTRACE_PLUGIN_SOURCE (
    set "SRC=%ARCHAEOTRACE_PLUGIN_SOURCE%"
) else (
    set "SRC=%~dp0ai_vectorizer"
)

if defined ARCHAEOTRACE_QGIS_PLUGINS_DIR (
    set "PLUGINSDIR=%ARCHAEOTRACE_QGIS_PLUGINS_DIR%"
) else (
    set "PLUGINSDIR=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins"
)
set "DEST=%PLUGINSDIR%\ai_vectorizer"

echo ===================================================
echo  ArchaeoTrace Dev Link Setup
echo ===================================================
echo Source: %SRC%
echo Destination: %DEST%
echo.

if not exist "%SRC%\metadata.txt" (
    echo [ERROR] Plugin source is missing metadata.txt.
    exit /b 1
)

if not exist "%PLUGINSDIR%" (
    mkdir "%PLUGINSDIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the QGIS plugins directory.
        exit /b 1
    )
)

if exist "%DEST%" (
    echo [ERROR] Refusing to replace an existing plugin directory or junction.
    echo Move or remove "%DEST%" explicitly, then run this script again.
    exit /b 1
)

echo [INFO] Creating directory junction...
mklink /J "%DEST%" "%SRC%"
if errorlevel 1 (
    echo [ERROR] Failed to create the link. Check directory permissions.
    exit /b 1
)

if not exist "%DEST%\metadata.txt" (
    echo [ERROR] The link command returned success, but the plugin is not readable.
    exit /b 1
)

echo [SUCCESS] Link created. QGIS will read directly from the source folder.
exit /b 0
