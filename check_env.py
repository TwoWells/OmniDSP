# check_env.py
import os
import sys
import importlib.util

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 20 + " sys.path " + "-" * 20)
for p in sys.path:
    print(p)

print("\n" + "-" * 20 + " os.environ['PATH'] " + "-" * 20)
paths = os.environ.get('PATH', '').split(';')
for p in paths:
    print(p)

# --- Try adding DLL directories directly ---
dll_paths_added = False
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    mkl_bin_dir = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin"
    compiler_bin_dir = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"

    print("\n" + "-" * 20 + " Adding DLL Dirs " + "-" * 20)
    if os.path.exists(mkl_bin_dir):
        try:
            os.add_dll_directory(mkl_bin_dir)
            print(f"Added: {mkl_bin_dir}") 
            dll_paths_added = True
        except OSError as e:
             print(f"ERROR adding MKL DLL dir {mkl_bin_dir}: {e}")

    if os.path.exists(compiler_bin_dir):
         try:
            os.add_dll_directory(compiler_bin_dir)
            print(f"Added: {compiler_bin_dir}")
            dll_paths_added = True
         except OSError as e:
             print(f"ERROR adding Compiler DLL dir {compiler_bin_dir}: {e}")

    if not dll_paths_added:
         print("WARNING: Failed to add required oneAPI DLL directories.")
    else:
         print("Successfully added required DLL directories.")

print("\n" + "-" * 20 + " Attempting Import " + "-" * 20)
try:
    # Find the specific .pyd file location (adjust if needed)
    # This assumes running from the 'build' directory
    spec = importlib.util.find_spec("omnidsp_py") 
    if spec and spec.origin:
        print(f"Found omnidsp_py spec at: {spec.origin}")
        # Attempt the import again now that DLL paths are added
        import omnidsp_py 
        print("\nSUCCESS: import omnidsp_py worked!")
        print("\n--- dir(omnidsp_py) ---")
        print(dir(omnidsp_py)) # Check if bindings are visible
    else:
        print("\nERROR: Could not find omnidsp_py module specification.")
        print("Is the script being run from the correct directory (e.g., 'build')?")
        print("Is PYTHONPATH set correctly if running from elsewhere?")

except ImportError as e:
    print(f"\nERROR: ImportError occurred: {e}")
    # Check if it's the specific DLL error again
    if "DLL load failed" in str(e):
         print("\nNOTE: This is the DLL load failure, indicating dependencies are still not found/resolved.")
    else:
         print("\nNOTE: This might be a different import error.")

except Exception as e:
    print(f"\nERROR: Unexpected error: {e}")