@echo off
TITLE emo run
echo=
echo  -----------选择自动检测还是手动检测--------------
echo=
echo （自动选a，每3秒做一次判断，手动选m，按s键进行判断）
echo=
choice /C am /M "auto, manual"
if %errorlevel% == 2 goto manual 
if %errorlevel% == 1 goto auto
 
:auto
python test_auto.py
goto end

:manual
python test_manual.py
goto end

:end
pause