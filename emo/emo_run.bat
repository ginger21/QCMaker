@echo off
TITLE emo run
echo=
echo  -----------ѡ���Զ���⻹���ֶ����--------------
echo=
echo ���Զ�ѡa��ÿ3����һ���жϣ��ֶ�ѡm����s�������жϣ�
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