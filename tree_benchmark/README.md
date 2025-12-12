### Generating Cubed-Sphere (CS) Meshes with TempestRemap

The cubed-sphere meshes are generated using TempestRemap’s `GenerateCSMesh` tool.

On Perlmutter, we use:

```bash
GenerateCSMesh --res <RES> --file <output.g> --alt
```

Where `RES` is the cubed-sphere resolution (cells per cube edge).  
In this benchmark, we use:

- `RES = 184`
- `RES = 369`
- `RES = 922`
- `RES = 1844`
- `RES = 4610`

(these correspond roughly to ~50 km, 25 km, 10 km, 5 km, and 2 km grids).
