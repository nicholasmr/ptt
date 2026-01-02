# Polarimetric traveltime tomography model

Model code and scripts for reproducing the plots of Rathmann (2026, *in review*).

<img src="https://raw.githubusercontent.com/nicholasmr/ptt/refs/heads/main/model.png?token=GHSAT0AAAAAACPHSIBFXYYTXVJSR5EHLU7C2KX5ZVA" alt="model" width="400px"> 

## What is it? 📡
- A polarimetric common midpoint (CMP) framework for ice sheets that accounts for the dielectric anisotropy arising from preferred crystal orientations and preferred bubble shapes in firn, including refractive bending due to density variations.
- Uses a Maxwell–Garnett effective medium formulation coupled to a single stage Herron–Langway density model with a power-law bubble eccentricity profile. 

## What can it do? 🧊
- Model the dieletric behavioir of ice and firn. 
- Model travel times along oblique ray paths. 
- Solve an inverse problem to robustly recover the fabric eigenvalue profile and bubble close-off depth from a limited set of CMP offsets, even in the presence of noise.
