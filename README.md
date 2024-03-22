# PhysicalFFT.jl

[![codecov](https://codecov.io/gh/JuliaAstroSim/PhysicalFFT.jl/graph/badge.svg?token=AtZqmsUQEj)](https://codecov.io/gh/JuliaAstroSim/PhysicalFFT.jl)

FFT solvers in Julia language

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

## Examples

```julia
using PhysicalFFT
using PhysicalMeshes

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