SPECTRAL BAND SPLITTER - README
================================

What this program does
----------------------
This tool loads a normal RGB image or a CR3 raw image, approximates it as a stack of spectral grayscale bands, and then reconstructs an image from those bands using editable spectral sensitivity curves and film-style characteristic curves.

Main idea:
1. Load an image
2. Separate it into spectral bands
3. Save a spectral project
4. Load the project in the Combine tab
5. Edit the spectral sensitivity and density curves
6. Preview the result live
7. Export an image or a 3D LUT

Important limitation
--------------------
This is still an approximation.
A normal RGB image does not contain the full real spectrum of the scene, so the spectral reconstruction is estimated from the RGB data.
That means the result can be useful and visually convincing, but it is not a true spectrometer measurement.

Supported input formats
-----------------------
- PNG
- JPG / JPEG
- TIFF / TIF
- BMP
- CR3 (if rawpy is installed)

Main features
-------------
- Separate tab for RGB/RAW -> spectral bands
- Combine tab for spectral reconstruction and look design
- Editable Red / Green / Blue spectral sensitivity curves
- Editable Red / Green / Blue H&D / characteristic density curves
- CSV import/export for individual R/G/B curves
- Live reconstruction preview
- Preview brightness control that does not affect exports
- Noise cleanup and film grain shaping controls
- Multiple spectral band export formats
- Multiple final image export formats
- 3D LUT export (.cube)
- LUT input and output color space selection
- Scene illuminant prior for spectral separation
- Background threading for heavy operations

Files created by separation
---------------------------
When you run separation, the program creates a spectral project folder.
That folder usually contains:
- spectral_cube.npz
- source_reference_rgb.npz
- metadata.json
- bands/ (individual spectral band files)

Recommended folder usage
------------------------
Keep the whole project folder together.
Do not move only one file out of it.
The Combine tab expects the project data to stay intact.

Dependencies
------------
Python packages used by the tool include:
- numpy
- pillow
- matplotlib
- tifffile
- rawpy (optional but needed for CR3)
- tkinter (usually included with Python on Windows)

If CR3 loading does not work:
- install rawpy
- make sure LibRaw support is available through the wheel

If TIFF export does not work:
- install tifffile

Performance notes
-----------------
- Separation and export run in background threads so the UI should stay responsive.
- Full-resolution separation can still take time, especially with many bands.
- 31 bands is a good default.
- 61 bands is a higher-quality mode but slower and heavier.
- 128^3 LUT export can be large and slower than 32^3 or 64^3.

Color pipeline notes
--------------------
- The reconstruction path is film-style, using spectral sensitivity + H&D density mapping.
- Preview brightness is preview-only.
- TIFF 32-bit float and RAW linear .npy preserve the most linear detail.
- PNG and TIFF 16-bit are better for ready-to-view images.
- The LUT path uses the current spectral sensitivity curves, density curves, density x-range, visible wavelength range, and chosen LUT input/output modes.

LUT notes
---------
The program exports .cube 3D LUTs.
Make sure the LUT input color space in the app matches the color space of the footage or image you will apply the LUT to.
If the input assumption is wrong, the LUT can look too bright, too saturated, or shifted.

Recommended default settings
----------------------------
General work:
- Bands: 31
- Wavelength range: 380 to 720 nm
- Band export: TIFF 16-bit or TIFF 32-bit float
- Final export: TIFF 32-bit float for maximum retained detail

Fast testing:
- Bands: 21
- Smaller preview max side
- LUT size: 32 or 64

Higher quality:
- Bands: 61
- TIFF 32-bit float band export
- RAW linear .npy or TIFF 32-bit float final export

What the main controls mean
---------------------------
Separate tab:
- Start nm / End nm: wavelength range used for the spectral approximation
- Band count: how many spectral grayscale slices are created
- Smoothness: regularization that keeps the recovered spectrum smoother
- Energy regularization: stabilizes the inversion
- Preview max side: resolution used for preview images and preview cube
- Band export format: format of the saved individual spectral band files
- Scene illuminant: Auto or manual light-source prior used during separation

Combine tab:
- Visible from/to nm: wavelength range used during reconstruction
- Sensitivity log ceiling: top scale of the spectral sensitivity editor
- Density log ceiling: top scale of the density editor
- Characteristic-curve exposure axis in lux·s: displays density x-axis in lux-seconds instead of relative log exposure
- Final export format: output format of the reconstructed image
- 3D LUT size: cube size for LUT export
- LUT input mode: assumed input color space for LUT generation
- LUT output mode: output encoding / color space for LUT generation
- Preview brightness: preview-only brightness balancer
- Noise cleanup: reduces coarse digital-looking noise in preview/exported positive image
- Film grain: adds a finer grain layer after cleanup

Known practical limitations
---------------------------
- RGB to spectrum is underdetermined, so some film differences remain subtle.
- Landscapes in neutral daylight may not show huge differences between film stocks.
- Stronger differences usually appear in tungsten scenes, LED scenes, skin tones, saturated reds, or mixed lighting.
- Accurate spectral sensitivity curves help, but dye absorption behavior and scan transforms also matter for realism.

PyInstaller / EXE build
-----------------------
Use the build files that were generated for this project:
- spectral_tool_app_v18_fixed.spec
- build_spectral_tool_fixed.bat

Place them next to:
- spectral_tool_app_v18_fixed.py

Then run the batch file.
The build output is typically placed under:
- dist\SpectralBandSplitter\

Troubleshooting
---------------
If the app does not open:
- confirm Python version is compatible
- confirm tkinter works on your system
- confirm matplotlib TkAgg backend is available

If CR3 fails:
- install rawpy

If TIFF fails:
- install tifffile

If the preview looks too bright:
- reduce Preview brightness
- export TIFF 32-bit float or RAW linear .npy to inspect unclipped linear output

If the LUT looks wrong in Photoshop or another app:
- make sure LUT input mode matches the document or footage color space
- make sure the host app is not interpreting a Rec.2020 LUT as sRGB or vice versa

