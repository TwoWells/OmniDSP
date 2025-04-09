@echo off
setlocal EnableDelayedExpansion

:: --- Configuration ---
:: EDIT THIS PATH if needed, used only if ONEAPI_ROOT is not set (fallback to find setvars)
set "DEFAULT_ONEAPI_SETVARS_PATH=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
:: Path where MKL DLLs reside, needed if delvewheel cannot find them via PATH
set "MKL_DLL_PATH_HINT=C:/path/to/your/mkl/redist/bin"
:: --- End Configuration ---

:: --- Default Build Options ---
set "BUILD_TYPE=Release"
REM Options: auto, mkl, accelerate
set "BACKEND_PREF=auto"
REM Options: ON, OFF
set "RUN_TESTS=ON"
REM Options: ON, OFF
set "BUILD_PYTHON=OFF"
set "INSTALL_PREFIX="
set "CMAKE_GENERATOR="
set "VENV_DIR=venv_build" REM Use a distinct venv name

:: --- Argument Parsing ---
:ParseArgsLoop
if "%~1"=="" goto EndParseArgs
if /I "%~1"=="-t" ( set "BUILD_TYPE=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--type" ( set "BUILD_TYPE=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="-b" ( set "BACKEND_PREF=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--backend" ( set "BACKEND_PREF=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--no-tests" ( set "RUN_TESTS=OFF" & shift & goto ParseArgsLoop )
if /I "%~1"=="--python" ( set "BUILD_PYTHON=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="-p" ( set "INSTALL_PREFIX=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--prefix" ( set "INSTALL_PREFIX=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="-G" ( set "CMAKE_GENERATOR=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="-h" ( goto PrintUsage )
if /I "%~1"=="--help" ( goto PrintUsage )
echo Unknown option: %1
goto PrintUsage
goto ParseArgsLoop
:EndParseArgs

:: Validate Build Type
if /I not "%BUILD_TYPE%"=="Release" if /I not "%BUILD_TYPE%"=="Debug" (
  echo Error: Invalid build type '%BUILD_TYPE%'. Use Release or Debug.
  exit /b 1
)
:: Validate Backend Preference
if /I not "%BACKEND_PREF%"=="auto" if /I not "%BACKEND_PREF%"=="mkl" if /I not "%BACKEND_PREF%"=="accelerate" (
  echo Error: Invalid backend preference '%BACKEND_PREF%'. Use auto, mkl, or accelerate.
  exit /b 1
)
:: Validate Python Build flag
if /I not "%BUILD_PYTHON%"=="ON" if /I not "%BUILD_PYTHON%"=="OFF" (
  echo Error: Invalid python build flag '%BUILD_PYTHON%'. Use ON or OFF.
  exit /b 1
)

echo --- Build Configuration ---
echo Build Type:       %BUILD_TYPE%
echo Backend Pref:     %BACKEND_PREF%
echo Build Python:     %BUILD_PYTHON%
echo Run Tests:        %RUN_TESTS%
if defined INSTALL_PREFIX (
  echo Install Prefix:   %INSTALL_PREFIX%
)
if defined CMAKE_GENERATOR (
  :: Keep quotes around generator name
  echo CMake Generator:  "%CMAKE_GENERATOR%"
)
echo -------------------------

:: --- Proactively attempt to set up MKL environment ---
echo Checking MKL environment setup...
if defined SETVARS_COMPLETED goto AlreadySet_MKL

  echo SETVARS_COMPLETED not set. Attempting to find setvars.bat...
  set "SETVARS_PATH="
  if not defined ONEAPI_ROOT goto TryDefaultSetvarsPath_MKL
  if not exist "%ONEAPI_ROOT%\setvars.bat" goto TryDefaultSetvarsPath_MKL
  echo Found setvars.bat via ONEAPI_ROOT.
  set "SETVARS_PATH=%ONEAPI_ROOT%\setvars.bat"
  goto FoundSetvarsPath_MKL
  :TryDefaultSetvarsPath_MKL
  echo Checking default path for setvars.bat...
  if not exist "%DEFAULT_ONEAPI_SETVARS_PATH%" goto SetvarsPathNotFound_MKL
  echo Found setvars.bat via default path.
  set "SETVARS_PATH=%DEFAULT_ONEAPI_SETVARS_PATH%"
  :FoundSetvarsPath_MKL
  if not defined SETVARS_PATH goto SetvarsPathNotFound_MKL
  goto CallSetvarsScript_MKL

:SetvarsPathNotFound_MKL
  echo Could not find setvars.bat via ONEAPI_ROOT or default path. Environment may not be set for MKL. Build might fail later if MKL is needed.
  goto EndSetvarsLogic_MKL

:AlreadySet_MKL
  echo SETVARS_COMPLETED environment variable already set ^('!SETVARS_COMPLETED!'^). Assuming environment is configured.
  goto EndSetvarsLogic_MKL

:CallSetvarsScript_MKL
  echo Calling Intel oneAPI environment script: "%SETVARS_PATH%"
  call "%SETVARS_PATH%" intel64
  echo Finished calling setvars.bat. Continuing script...
  :: Fall through to EndSetvarsLogic

:EndSetvarsLogic_MKL
echo -------------------------


:: --- Setup Python Environment (if requested) ---
if /I "%BUILD_PYTHON%"=="ON" (
    echo Setting up Python virtual environment...
    if exist "%VENV_DIR%" (
        echo Removing existing venv: %VENV_DIR%
        rmdir /S /Q "%VENV_DIR%"
        if errorlevel 1 ( echo Failed to remove existing venv. Check permissions. & exit /b 1)
    )
    echo Creating venv: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo Failed to create Python virtual environment. & exit /b 1 )

    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    if errorlevel 1 ( echo Failed to activate Python virtual environment. & exit /b 1 )

    echo Installing Python dependencies...
    pip install -r requirements.txt
    if errorlevel 1 ( echo Failed to install requirements.txt & exit /b 1 )
    if exist requirements-dev.txt (
        pip install -r requirements-dev.txt
        if errorlevel 1 ( echo Failed to install requirements-dev.txt & exit /b 1 )
    ) else (
        echo requirements-dev.txt not found, skipping.
    )
    echo Python environment ready.
    echo -------------------------
)


:: --- Clean and Create Build Directory ---
echo Cleaning and creating build directory...
if exist "build" (
  echo Removing existing build directory...
  rmdir /S /Q "build"
  if errorlevel 1 ( echo Failed to remove build directory. Check permissions. & exit /b 1)
)
echo Creating build directory...
mkdir "build"
if errorlevel 1 ( echo Failed to create build directory. & exit /b 1)
echo Changing into build directory...
cd "build"


:: --- Construct CMake Command ---
set "CMAKE_ARGS=../"
set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%"
set "CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_SHARED_LIBS=OFF"

:: Add Python build flag if enabled
if /I "%BUILD_PYTHON%"=="ON" (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DOMNIDSP_BUILD_PYTHON_BINDINGS=ON"
) else (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DOMNIDSP_BUILD_PYTHON_BINDINGS=OFF"
)


:: Set CMake flags based on user preference
if /I "%BACKEND_PREF%"=="mkl" (
  echo Setting CMake flag to prefer MKL...
  set "CMAKE_ARGS=%CMAKE_ARGS% -DFFT_PREFER_ONEMKL=ON"
) else if /I "%BACKEND_PREF%"=="accelerate" (
  echo Setting CMake flags to prefer Accelerate...
    set "CMAKE_ARGS=%CMAKE_ARGS% -DFFT_PREFER_ONEMKL=OFF"
    set "CMAKE_ARGS=%CMAKE_ARGS% -DFFT_PREFER_ACCELERATE=ON"
) else (
  echo Using CMake default backend preference ^(auto^)...
  set "CMAKE_ARGS=%CMAKE_ARGS% -DFFT_PREFER_ONEMKL=OFF"
)

if defined INSTALL_PREFIX (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INSTALL_PREFIX=\"%INSTALL_PREFIX%\""
)

if defined CMAKE_GENERATOR (
  set "CMAKE_ARGS=%CMAKE_ARGS% -G \"%CMAKE_GENERATOR%\""
)

:: --- Run CMake Configuration ---
echo Running CMake configure...
echo cmake %CMAKE_ARGS%
cmake %CMAKE_ARGS%
if errorlevel 1 ( echo CMake configuration failed. & cd .. & exit /b 1 )

:: --- Run CMake Build ---
echo Running CMake build...
cmake --build . --config %BUILD_TYPE% --parallel
if errorlevel 1 ( echo CMake build failed. & cd .. & exit /b 1 )

:: --- Run Tests ---
if /I "%RUN_TESTS%"=="ON" (
  echo Running tests...
  ctest -C %BUILD_TYPE% --output-on-failure
  if errorlevel 1 ( echo Tests failed. Check output above. Build artifacts remain. & cd .. & exit /b 1 )
  echo Tests passed.
) else (
  echo Skipping tests.
)

:: --- Build and Repair Python Wheel (if requested) ---
if /I "%BUILD_PYTHON%"=="ON" (
    echo -------------------------
    echo Building and repairing Python wheel...

    echo Returning to project root directory...
    cd ..
    if errorlevel 1 ( echo Failed to change directory to project root. & exit /b 1 )

    echo Cleaning previous Python build artifacts...
    if exist dist rmdir /S /Q dist
    if exist wheelhouse rmdir /S /Q wheelhouse

    echo Building wheel...
    pip wheel . --no-deps -w dist
    if errorlevel 1 ( echo pip wheel command failed. & exit /b 1 )

    echo Repairing wheel with delvewheel...
    REM Attempting to rely on PATH set by setvars.bat for MKL DLLs
    REM If this fails, uncomment and adjust the --add-path argument below
    REM delvewheel repair --add-path "%MKL_DLL_PATH_HINT%" -w wheelhouse dist/omnidsp_py-*.whl
    delvewheel repair -w wheelhouse dist/omnidsp_py-*.whl
    if errorlevel 1 (
      echo delvewheel repair failed.
      echo Possible reasons:
      echo  - delvewheel not installed ^(run pip install delvewheel^).
      echo  - MKL DLLs not found in PATH ^(set by setvars.bat^).
      echo  - Try uncommenting and setting --add-path in the script.
      exit /b 1
     )
    echo Repaired wheel created in wheelhouse/
    echo -------------------------
) else (
    REM If not building Python, just return to original directory from build/
     cd ..
)


echo Build script finished successfully.
endlocal
exit /b 0

:: --- PrintUsage should remain at the end for error cases ---
:PrintUsage
echo Usage: build.bat [options]
echo Options:
echo   -t, --type ^<TYPE^>       Build type: Release ^(default^), Debug
echo   -b, --backend ^<PREF^>    Backend preference: auto ^(default^), mkl, accelerate
echo       --python ^<ON|OFF^>   Build Python bindings and wheel (default: OFF)
echo       --no-tests          Disable running tests ^(default: ON^)
echo   -p, --prefix ^<PATH^>     Install prefix path ^(default: none^)
echo   -G ^<GENERATOR^>       Specify CMake generator ^(e.g., "Visual Studio 17 2022"^)
echo   -h, --help              Show this help message
exit /b 1