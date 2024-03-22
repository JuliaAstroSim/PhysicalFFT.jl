using Test
using AstroSimBase
using PhysicalMeshes
using PhysicalMeshes.PhysicalParticles
using PhysicalFFT

sol(x::Real) = sin(2*pi*x) + sin(32*pi*x) / 256
init_rho(x::Real) = -4*pi*pi*sin(2*pi*x) - 4*pi*pi*sin(32*pi*x)

sol(p::PVector2D) =  sin(2*pi*p.x) * sin(2*pi*p.y) + sin(32*pi*p.x) * sin(32*pi*p.y) / 256
init_rho(p::PVector2D) = -8 * pi * pi * sin(2*pi*p.x) * sin(2*pi*p.y) - 8 * pi * pi * sin(32*pi*p.x) * sin(32*pi*p.y)

sol(p::PVector) =  sin(2*pi*p.x) * sin(2*pi*p.y) * sin(2*pi*p.z) + sin(32*pi*p.x) * sin(32*pi*p.y) * sin(2*pi*p.z) / 256
init_rho(p::PVector) = -12 * pi * pi * sin(2*pi*p.x) * sin(2*pi*p.y) * sin(2*pi*p.z) - 12 * pi * pi * sin(32*pi*p.x) * sin(32*pi*p.y) * sin(32*pi*p.z)

L2norm(r) = sqrt(sum((r.^2)/prod(size(r))))

@testset "1D" begin
    function test_fft1D(Nx, boundary=Periodic())
        m = MeshCartesianStatic(;
            xMin = 0.0,
            xMax = 1.0,
            Nx = Nx - 1,
            NG = 0,
            dim = 1,
            boundary,
        )
        m.rho .= init_rho.(m.pos)
        fft_poisson!(m, m.rho, m.config.boundary)
        s = sol.(m.pos)
        r = m.phi .- s

        return L2norm(r)
    end

    @testset "Periodic" begin
        @test test_fft1D(8, Periodic()) < 0.5
        @test test_fft1D(128, Periodic()) < 0.016
    end

    @testset "Dirichlet" begin
        @test test_fft1D(8, Dirichlet()) < 0.7
        @test test_fft1D(128, Dirichlet()) < 0.031
    end
end

@testset "2D" begin
    function test_fft2D(Nx, boundary=Periodic())
        m = MeshCartesianStatic(;
            xMin = 0.0,
            yMin = 0.0,
            xMax = 1.0,
            yMax = 1.0,
            Nx = Nx - 1,
            Ny = Nx - 1,
            NG = 0,
            dim = 2,
            boundary,
        )
        m.rho .= init_rho.(m.pos)
        fft_poisson!(m, m.rho, m.config.boundary)
        s = sol.(m.pos)
        r = m.phi .- s

        return L2norm(r)
    end

    @testset "Periodic" begin
        @test test_fft2D(8, Periodic()) < 0.3
        @test test_fft2D(128, Periodic()) < 0.012
    end

    @testset "Dirichlet" begin
        @test test_fft2D(8, Dirichlet()) < 0.3
        @test test_fft2D(128, Dirichlet()) < 0.02
    end
end

@testset "3D" begin
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

    @testset "Periodic" begin
        @test test_fft3D(8, Periodic()) < 0.15
        @test test_fft3D(16, Periodic()) < 0.41
    end

    @testset "Dirichlet" begin
        @test test_fft3D(8, Dirichlet()) < 0.14
        @test test_fft3D(16, Dirichlet()) < 0.40
    end
end