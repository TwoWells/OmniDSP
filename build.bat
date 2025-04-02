@echo off
setlocal EnableDelayedExpansion

:: --- Configuration ---
:: EDIT THIS PATH if needed, used only if ONEAPI_ROOT is not set (fallback to find setvars)
set "DEFAULT_ONEAPI_SETVARS_PATH=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
:: --- End Configuration ---

:: --- Default Build Options ---
set "BUILD_TYPE=Release"
REM Options: auto, mkl, accelerate
set "BACKEND_PREF=auto"
REM Options: ON, OFF
set "RUN_TESTS=ON"
set "INSTALL_PREFIX="
set "CMAKE_GENERATOR="

:: --- Argument Parsing ---
:ParseArgsLoop
if "%~1"=="" goto EndParseArgs
if /I "%~1"=="-t" ( set "BUILD_TYPE=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--type" ( set "BUILD_TYPE=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="-b" ( set "BACKEND_PREF=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--backend" ( set "BACKEND_PREF=%~2" & shift & shift & goto ParseArgsLoop )
if /I "%~1"=="--no-tests" ( set "RUN_TESTS=OFF" & shift & goto ParseArgsLoop )
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

echo --- Build Configuration ---
echo Build Type:       %BUILD_TYPE%
echo Backend Pref:     %BACKEND_PREF%
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
if defined SETVARS_COMPLETED goto AlreadySet

  echo SETVARS_COMPLETED not set. Attempting to find setvars.bat...
  set "SETVARS_PATH="
  if not defined ONEAPI_ROOT goto TryDefaultSetvarsPath_L1
  if not exist "%ONEAPI_ROOT%\setvars.bat" goto TryDefaultSetvarsPath_L1
  echo Found setvars.bat via ONEAPI_ROOT.
  set "SETVARS_PATH=%ONEAPI_ROOT%\setvars.bat"
  goto FoundSetvarsPath_L1
  :TryDefaultSetvarsPath_L1
  echo Checking default path for setvars.bat...
  if not exist "%DEFAULT_ONEAPI_SETVARS_PATH%" goto SetvarsPathNotFound_L1
  echo Found setvars.bat via default path.
  set "SETVARS_PATH=%DEFAULT_ONEAPI_SETVARS_PATH%"
  :FoundSetvarsPath_L1
  if not defined SETVARS_PATH goto SetvarsPathNotFound_L1
  goto CallSetvarsScript

:SetvarsPathNotFound_L1
  echo Could not find setvars.bat via ONEAPI_ROOT or default path. Environment may not be set for MKL. Build might fail later if MKL is needed.
  goto EndSetvarsLogic

:AlreadySet
  echo SETVARS_COMPLETED environment variable already set ^('!SETVARS_COMPLETED!'^). Assuming environment is configured.
  goto EndSetvarsLogic

:CallSetvarsScript
  echo Calling Intel oneAPI environment script: "%SETVARS_PATH%"
  call "%SETVARS_PATH%" intel64
  echo Finished calling setvars.bat. Continuing script...
  :: Fall through to EndSetvarsLogic

:EndSetvarsLogic
echo -------------------------


:: --- Clean and Create Build Directory ---
echo Cleaning and creating build directory...
if not exist "build" goto BuildDirAbsent

  echo Removing existing build directory...
  rmdir /S /Q "build"
  if errorlevel 1 goto RmdirFailed
  :: Fall through to create directory

:BuildDirAbsent
  echo Creating build directory...
  mkdir "build"
  if errorlevel 1 goto MkdirFailed
  echo Changing into build directory...
  cd "build"
  goto PostDirHandling

:RmdirFailed
  echo Failed to remove build directory. Check permissions.
  exit /b 1

:MkdirFailed
  echo Failed to create build directory.
  exit /b 1

:PostDirHandling
:: Now inside the build directory


:: --- Construct CMake Command ---
set "CMAKE_ARGS=../"
set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%"
set "CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_SHARED_LIBS=OFF"


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
  if errorlevel 1 ( echo Tests failed. & cd .. & exit /b 1 )
) else (
  echo Skipping tests.
)

:: --- Return to original directory ---
cd ..

echo Build script finished successfully.
endlocal
exit /b 0

:: --- PrintUsage should remain at the end for error cases ---
:PrintUsage
echo Usage: build.bat [options]
echo Options:
echo   -t, --type ^<TYPE^>      Build type: Release ^(default^), Debug
echo   -b, --backend ^<PREF^>   Backend preference: auto ^(default^), mkl, accelerate
echo       --no-tests           Disable running tests ^(default: ON^)
echo   -p, --prefix ^<PATH^>    Install prefix path ^(default: none^)
echo   -G ^<GENERATOR^>       Specify CMake generator ^(e.g., "Visual Studio 17 2022"^)
echo   -h, --help             Show this help message
exit /b 1