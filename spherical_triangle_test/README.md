# Ill-Conditioned Spherical Triangle Inputs

This directory doesn't contains the full copy of the input dataset because the generated datasets can be large.  
Users may regenerate the inputs using the steps below.

---

## Generate Ill-Conditioned Triangles

The executable `generate_ill_triangles` produces the small-angle test triangles used in the benchmark, similar to how ARPIST generates its ill-conditioned cases (https://github.com/numgeom/ARPIST).  
After building the project, run the following to generate the input:


```bash
# build
cd /Regrid_Benchmark/src/spherical_triangle_test

cmake -S . -B <build directory> -DCMAKE_BUILD_TYPE=Release
cmake --build <build directory> -j


#    N = polar-angle subdivisions
#    M = phi subdivisions
generate_ill_triangles <N> <M>

# Example (default resolution):
./generate_ill_triangles 60 360
```
The generation will write the following file into the specified output directory
```bash
<output directory>/ill_triangles_<M>_<N>.csv
```

## Generate Icosahedral Triangles

The icosahedral triangle meshes used in this study were generated using  
`GenerateICOMesh`, the standard `TempestRemap` mesh generator.

To reproduce the meshes, run:

```bash
GenerateICOMesh --res <r> --file <output.nc> --out_format Netcdf4
```
where `r` is the icosahedral refinement level.

The meshes in this benchmark correspond to the following **r** values:

- `r = 7`
- `r = 14`
- `r = 20`
- `r = 29`

These values produce triangle counts close to the commonly used  
**1k / 4k / 8k / 16k** element sizes.