# Accurate-and-Robust-Geometric-Algorithms-for-Regridding-on-the-Sphere-benchmark


This repository contains the code and input data used for all benchmarks in the manuscript  
**“Accurate and Robust Geometric Algorithms for Regridding on the Sphere.”**  

If you have questions, feel free to contact: **hyvchen@ucdavis.edu**.
## Benchmark components

The project is organized into four main benchmarks:

### 1. `GCAGCA_intersection/`
Accuracy benchmark for different great-circle–arc (GCA × GCA) intersection algorithms.  
This includes random ill-conditioned arc pairs as well as wide-span cases.  
Outputs: intersection error statistics and comparisons across methods.

### 2. `GCA_Angle_test/`
Accuracy and performance benchmark for different ways of computing great-circle arc length (i.e., spherical angle).  
Covers edge cases such as nearly antipodal points and very small angles.

### 3. `spherical_triangle_test/`
Accuracy benchmark for spherical triangle area evaluation.  
This includes:
- Eriksson formula  
- ARPIST / TempestRemap / YAC quadratures  
- Automatically generated ill-conditioned triangles  
- Automatically generated ICO meshes for reall-world application tests

### 4. `tree_benchmark/`
Performance benchmark comparing KDTree and BallTree on large UGRID meshes.  

## What is included

- Input data (or scripts to generate them).  
  For cases where the dataset is too large to store directly, each subdirectory has its own `README.md` explaining how to generate the required input files.


- Benchmark scripts for each experiment.  
  Accuracy benchmarks use Mathematica for the reference computations.  
  Performance benchmarks are standalone C++/Python and mainly depend on Eigen (and basic sklearn for the tree benchmark).
