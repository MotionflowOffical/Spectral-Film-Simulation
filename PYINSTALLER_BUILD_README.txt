PyInstaller build files for spectral_tool_app_v18_fixed.py

Files:
- spectral_tool_app_v18_fixed.spec
- build_spectral_tool.bat

How to use on Windows:
1. Put both files in the same folder as spectral_tool_app_v18_fixed.py
2. Double-click build_spectral_tool.bat
3. The built app will be created in:
   dist\SpectralBandSplitter\

Notes:
- The build script installs/updates: pyinstaller, numpy, pillow, matplotlib, tifffile, rawpy
- rawpy and tifffile are bundled only if they are installed in the Python environment used for the build
- The spec is configured as a folder build, not --onefile. That is usually more reliable for Tk/matplotlib/rawpy apps
- The executable name is: SpectralBandSplitter.exe
