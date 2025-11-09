# PyTissueOptics API Reference

!!! warning "API Under Construction"
    The API reference is **currently under construction**.
    Some classes, functions, and modules may be missing or incomplete.

This page provides an overview of the main user-facing classes and functions, as well as links to the complete API reference.

---

## Main User API

These are the main classes and objects you will interact with as a user:

- **Solids** — 3D geometric objects for defining scenes.  
- **ScatteringMaterial** — Defines optical properties of solids.
- **Sources** — Light sources for simulations.  
- **ScatteringScene** — Define a group of solids for Monte-Carlo simulations.
- **EnergyLogger** — Records photon interactions during simulations.
- **Viewer** — Visualize simulation results.

> These classes cover the core workflow for building, simulating, and visualizing light propagation.

---

## Complete API Reference

Note that the code is separated in two main modules: 

### Monte-Carlo light propagation

- [rayscattering](rayscattering.md) — Monte-Carlo photon propagation, statistics, and visualization.

### Core Graphics Framework

- [scene](scene.md) — Core 3D graphics framework with raytracing and visualization.
