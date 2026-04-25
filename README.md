# Polarimetric traveltime tomography model

Model code and scripts for reproducing the plots of Rathmann (2026, RSPA).

<img src="https://raw.githubusercontent.com/nicholasmr/ptt/refs/heads/main/model.png" alt="model" width="450px"> 

## What is it? 📡
- A polarimetric common midpoint (CMP) framework for ice sheets that accounts for the dielectric anisotropy arising from preferred crystal orientations and preferred bubble shapes in firn, including refractive bending due to density variations.
- Uses a Maxwell–Garnett effective medium formulation coupled to a single stage Herron–Langway density model with a power-law bubble eccentricity profile. 

## What can it do? 🧊
- Model the dieletric behavioir of ice and firn. 
- Model travel times along oblique ray paths. 
- Solve an inverse problem to robustly recover the fabric eigenvalue profile and bubble close-off depth from CMP traveltime data (even in the presence of noise).

