@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
rc /nologo app.rc
if errorlevel 1 echo rc failed
cl /nologo /utf-8 /O2 /W3 /DUNICODE /D_UNICODE launcher.c app.res /link /SUBSYSTEM:WINDOWS /OUT:CrispFrame.exe
dir CrispFrame.exe
