# PhysicalFFT.jl

[![codecov](https://codecov.io/gh/JuliaAstroSim/PhysicalFFT.jl/graph/badge.svg?token=AtZqmsUQEj)](https://codecov.io/gh/JuliaAstroSim/PhysicalFFT.jl)

FFT PDE solvers in Julia language.

WARNING: *This package is under development!!!*

## Installation

```julia
]add PhysicalFDM
```

or

```julia
using Pkg; Pkg.add("PhysicalFDM")
```

or

```julia
using Pkg; Pkg.add("https://github.com/JuliaAstroSim/PhysicalFDM.jl")
```

To test the Package:
```julia
]test PhysicalFDM
```

## User Guide

This package is extracted from [AstroNbodySim.jl](https://github.com/JuliaAstroSim/AstroNbodySim.jl). You may find more advanced examples there.

You can also reference the usage of finite differencing solver [PhysicalFDM.jl](https://github.com/JuliaAstroSim/PhysicalFDM.jl).

### Comparison of `PhysicalFDM.jl` and `PhysicalFFT.jl`

| Feature | `PhysicalFDM.jl` | `PhysicalFFT.jl` |
| :--: | :--: | :--: |
| 1D Poisson | √ | √ |
| 2D Poisson | √ | √ |
| 3D Poisson | √ | √ |
| Periodic BCs | √ | √ |
| Dirichlet BCs | √ | √ |
| Vacuum BCs | √ | × |
| GPU | √ | √ |

The main disadvantage of `PhysicalFDM.jl` is that the computational complexity (the matrix size) scales with $M^{d^d}$,
where $M$ is the mesh size in each direction and `d` is the dimension of the problem.
Consequently, for meshes $\ge 16^3$, the memory usage ($\ge$ 64GB) and computation time ($\ge$ 10 hrs) are not affordable.

`PhysicalFFT.jl` supports resolution of $512^3$ with minimum effort: 10 sec on both GPU (shared GPU memory is used) and CPU.
For meshes smaller than $256^3$, the computation time on GPU is 1~4 orders of magnitude smaller than on CPU.
However, the vacuum boundary conditions, which are necessary for isolated gravitational systems,
are not yet supported in `PhysicalFFT.jl`.
Nevertheless, the errors from periodic boundary conditions are tolerable if the simulation box is sufficiently large compared to the system's length scale.

## Examples

### Solve Poisson equation

$$\Delta u = f$$

```julia
using PhysicalFFT
using PhysicalMeshes
using PhysicalMeshes.Particles

sol(p::PVector) =  sin(2*pi*p.x) * sin(2*pi*p.y) * sin(2*pi*p.z) + sin(32*pi*p.x) * sin(32*pi*p.y) * sin(2*pi*p.z) / 256
init_rho(p::PVector) = -12 * pi * pi * sin(2*pi*p.x) * sin(2*pi*p.y) * sin(2*pi*p.z) - 12 * pi * pi * sin(32*pi*p.x) * sin(32*pi*p.y) * sin(32*pi*p.z)

function test_fft3D(Nx, boundary=Periodic())
    m = MeshCartesianStatic(;
        xMin = 0.0,
        yMin = 0.0,
        zMin = 0.0,
        xMax = 1.0,
        yMax = 1.0,
        zMax = 1.0,
        Nx = Nx - 1,
        Ny = Nx - 1,
        Nz = Nx - 1,
        NG = 0,
        dim = 3,
        boundary,
    )
    m.rho .= init_rho.(m.pos)
    fft_poisson!(m, m.rho, m.config.boundary)
    s = sol.(m.pos)
    r = m.phi .- s

    return L2norm(r)
end

test_fft3D(8, Periodic())
```

## TODO list

- [ ] Vacuum boundary conditions for isolated system
- [ ] Test GPU
- [ ] FFT spectral solver for Schrödinger-Poisson equation (SPE)

## Package ecosystem

- Basic data structure: [PhysicalParticles.jl](https://github.com/JuliaAstroSim/PhysicalParticles.jl)
- File I/O: [AstroIO.jl](https://github.com/JuliaAstroSim/AstroIO.jl)
- Initial Condition: [AstroIC.jl](https://github.com/JuliaAstroSim/AstroIC.jl)
- Parallelism: [ParallelOperations.jl](https://github.com/JuliaAstroSim/ParallelOperations.jl)
- Trees: [PhysicalTrees.jl](https://github.com/JuliaAstroSim/PhysicalTrees.jl)
- Meshes: [PhysicalMeshes.jl](https://github.com/JuliaAstroSim/PhysicalMeshes.jl)
- Finite differencing solver [PhysicalFDM.jl](https://github.com/JuliaAstroSim/PhysicalFDM.jl)
- FFT solver [PhysicalFFT.jl](https://github.com/JuliaAstroSim/PhysicalFFT.jl)
- Plotting: [AstroPlot.jl](https://github.com/JuliaAstroSim/AstroPlot.jl)
- Simulation: [AstroNbodySim.jl](https://github.com/JuliaAstroSim/AstroNbodySim.jl)