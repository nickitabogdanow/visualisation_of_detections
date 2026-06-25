@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

for /f "usebackq delims=" %%i in (`python -c "import plotly, pathlib; print(pathlib.Path(plotly.__path__[0]) / 'package_data' / 'plotly.min.js')"`) do set PLOTLY_JS=%%i

echo Building executable...
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --console ^
  --name IDFVisualisation ^
  --add-data "templates;templates" ^
  --add-data "%PLOTLY_JS%;." ^
  --hidden-import main ^
  --collect-data plotly ^
  --collect-submodules uvicorn ^
  start.py

if not exist "dist\data" mkdir "dist\data"

echo.
echo Done.
echo Executable: dist\IDFVisualisation.exe
echo Put CSV files into: dist\data
echo Run dist\IDFVisualisation.exe and open the printed URL in a browser.

endlocal
