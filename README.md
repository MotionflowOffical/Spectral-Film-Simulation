## Overview

The basic workflow is:

1. Load an image
2. Separate it into spectral bands
3. Save a spectral project
4. Load that project in the **Combine** tab
5. Edit the spectral sensitivity and density curves
6. Preview the result live
7. Export a reconstructed image or a 3D LUT

## Important Limitation

This tool is still an **approximation**.

A normal RGB image does not contain the full real spectrum of the scene, so the spectral reconstruction is estimated from RGB data. That means the result can be visually convincing and useful for experimentation, but it is **not a true spectrometer measurement**.

## Supported Input Formats

- PNG
- JPG / JPEG
- TIFF / TIF
- BMP
- CR3 *(when `rawpy` is installed)*

## Main Features

- **Separate** tab for RGB/RAW → spectral band separation
- **Combine** tab for spectral reconstruction and look design
- Editable **Red / Green / Blue** spectral sensitivity curves
- Editable **Red / Green / Blue** H&D / characteristic density curves
- CSV import/export for individual **R / G / B** curves
- Live reconstruction preview
- Preview brightness control that does **not** affect exports
- Noise cleanup and film grain shaping controls
- Multiple spectral band export formats
- Multiple final image export formats
- 3D LUT export (`.cube`)
- LUT input and output color space selection
- Scene illuminant prior for spectral separation
- Background threading for heavy operations

## Files Created by Separation

When separation runs, the program creates a **spectral project folder**. That folder usually contains:

- `spectral_cube.npz`
- `source_reference_rgb.npz`
- `metadata.json`
- `bands/` *(individual spectral band files)*

## Recommended Folder Usage

Keep the **entire project folder together**.

Do not move only one file out of it. The **Combine** tab expects the project data to remain intact.

## Dependencies

This tool uses the following Python packages:

- `numpy`
- `pillow`
- `matplotlib`
- `tifffile`
- `rawpy` *(optional, required for CR3 support)*
- `tkinter` *(usually included with Python on Windows)*

### CR3 loading issues

- Install `rawpy`
- Make sure the installed wheel includes usable LibRaw support

### TIFF export issues

- Install `tifffile`

## Performance Notes

- Separation and export run in background threads, so the UI should stay responsive
- Full-resolution separation can still take time, especially with many bands
- **31 bands** is a good default
- **61 bands** is a higher-quality mode, but slower and heavier
- **128³ LUT export** can be much slower and heavier than **32³** or **64³**

## Color Pipeline Notes

- The reconstruction path is film-style, using **spectral sensitivity + H&D density mapping**
- Preview brightness is **preview-only**
- `TIFF 32-bit float` and `RAW linear .npy` preserve the most linear detail
- PNG and `TIFF 16-bit` are better for ready-to-view images
- The LUT path uses the current:
  - spectral sensitivity curves
  - density curves
  - density x-range
  - visible wavelength range
  - LUT input/output modes

## LUT Notes

The program exports **`.cube` 3D LUTs**.

Make sure the LUT input color space in the app matches the color space of the footage or image the LUT will be applied to.

If the input assumption is wrong, the LUT can appear:

- too bright
- too saturated
- color shifted

## Recommended Default Settings

### General work

- **Bands:** 31
- **Wavelength range:** 380 to 720 nm
- **Band export:** `TIFF 16-bit` or `TIFF 32-bit float`
- **Final export:** `TIFF 32-bit float` for maximum retained detail

### Fast testing

- **Bands:** 21
- Smaller preview max side
- **LUT size:** 32 or 64

### Higher quality

- **Bands:** 61
- `TIFF 32-bit float` band export
- `RAW linear .npy` or `TIFF 32-bit float` final export

## Main Controls

### Separate Tab

- **Start nm / End nm**  
  Wavelength range used for the spectral approximation

- **Band count**  
  Number of spectral grayscale slices created

- **Smoothness**  
  Regularization that keeps the recovered spectrum smoother

- **Energy regularization**  
  Stabilizes the inversion

- **Preview max side**  
  Resolution used for preview images and the preview cube

- **Band export format**  
  Format used for the saved individual spectral band files

- **Scene illuminant**  
  Auto or manual light-source prior used during separation

### Combine Tab

- **Visible from / to nm**  
  Wavelength range used during reconstruction

- **Sensitivity log ceiling**  
  Top scale of the spectral sensitivity editor

- **Density log ceiling**  
  Top scale of the density editor

- **Characteristic-curve exposure axis in lux·s**  
  Displays the density x-axis in lux-seconds instead of relative log exposure

- **Final export format**  
  Output format of the reconstructed image

- **3D LUT size**  
  Cube size for LUT export

- **LUT input mode**  
  Assumed input color space for LUT generation

- **LUT output mode**  
  Output encoding / color space for LUT generation

- **Preview brightness**  
  Preview-only brightness balancer

- **Noise cleanup**  
  Reduces coarse digital-looking noise in the preview/exported positive image

- **Film grain**  
  Adds a finer grain layer after cleanup

## Known Practical Limitations

- RGB → spectrum is underdetermined, so some film differences remain subtle
- Landscapes in neutral daylight may not show very large differences between film stocks
- Stronger differences usually appear in:
  - tungsten scenes
  - LED scenes
  - skin tones
  - saturated reds
  - mixed lighting
- Accurate spectral sensitivity curves help, but **dye absorption behavior** and **scan transforms** also matter for realism

## PyInstaller / EXE Build

Use the generated build files:

- `spectral_tool_app_v18_fixed.spec`
- `build_spectral_tool_fixed.bat`

Place them next to:

- `spectral_tool_app_v18_fixed.py`

Then run the batch file.

Build output is usually placed under:

- `dist\SpectralBandSplitter\`

## Troubleshooting

### App does not open

- Confirm the Python version is compatible
- Confirm `tkinter` works on your system
- Confirm the matplotlib `TkAgg` backend is available

### CR3 loading fails

- Install `rawpy`

### TIFF export fails

- Install `tifffile`

### Preview looks too bright

- Reduce **Preview brightness**
- Export `TIFF 32-bit float` or `RAW linear .npy` to inspect unclipped linear output

### LUT looks wrong in Photoshop or another app

- Make sure **LUT input mode** matches the document or footage color space
- Make sure the host application is not interpreting a **Rec.2020 LUT** as **sRGB**, or the reverse

## Summary

Spectral Band Splitter is a spectral-approximation and film-reconstruction tool designed for experimentation, curve-based look design, and LUT generation from RGB or CR3 input.

It is most useful when treated as:

- a **spectral look-development tool**
- a **film-style reconstruction tool**
- an **experimental color pipeline**

rather than a physically exact spectrometry system.
