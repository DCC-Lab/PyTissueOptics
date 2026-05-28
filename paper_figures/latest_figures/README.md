# Paper figures — latest versions

Each subfolder is one figure: its script(s), the rendered outputs (PDF/PNG), and (for the
brain figure) the data/dependency it needs. Run any script from inside its own subfolder
with the project venv active:

```
source /Users/williamboucher/Documents/pytissue_paperfigures/PyTissueOptics/.venv/bin/activate
cd <subfolder>
MPLBACKEND=Agg python <script>.py
```

| Subfolder | Script | Outputs |
|---|---|---|
| `fig1_pencil_beam/` | `fig1_combined_refraction_vs_scattering.py` | refraction / scattering / side-by-side PNGs (matched Mayavi camera + FOV, white bg; Snell/Fresnel/Beer–Lambert verification printed) |
| `fig2_lens/` | `fig2_lens_focusing_traces.py` | `fig_lens_psf_v2.pdf`, `fig_lens_focal_spot_investigation.png`, `fig_lens_traces_3d.png` (native traces, white-bg 3D, 576-tri ring-caustic study, smoothed vs high-poly-flat) |
| `fig_caustic/` | `fig_paper_caustic_v2.py` | `fig_paper_caustic_v2.pdf/.png` (native-trace meridional caustic; thick dashed nominal-f; no grid) |
| `fig2d_chromatic/` | `fig2d_chromatic_aberration_v2.py` | `fig2d_chromatic_aberration_v2.pdf/.png` (restyled: enlarged text, dash-dot nominal-f, no grid) |
| `fig3_4f/` | `fig3_4f_system_v2.py` | `fig3_4f_rays_v2.*`, `fig3_4f_3d_v2.png`, `fig3_4f_combined_v2.*` (native-trace meridional + white-bg left-to-right 3D, axially-aligned stack) |
| `fig4b_brain/` | `fig4b_anatomy_v2.py` | `fig4b_anatomy_v2.pdf/.png`, `fig4b_cortex_photons.png` (sampling volume + 3D cortical surface; rendered at 250k photons) |

## `fig4b_brain/` notes
- Self-contained: it ships `fig4b_sampling_volume_comparison.py` (its sim/raster dependency)
  and `MMC_Collins_Atlas_Mesh_Version_2L.mat` (Colin27 4-layer surface mesh), and auto-detects
  the local mesh, so it runs from this folder.
- Default `FIG4B_N=30000` for quick runs; the committed figure used `FIG4B_N=250000`. Render
  tunables (camera azimuth, cortex/scalp opacity, detector, cloud points) are at the top.

## Notes
- 3D panels are real Mayavi/VTK renders embedded into matplotlib.
- The originals these supersede live in `../previous_figures/`.
