module PhysicalFFT

using FFTW
using CUDA
using PrecompileTools

using Unitful

using AstroSimBase
using PhysicalMeshes
using PhysicalMeshes.PhysicalParticles


export fft_poisson!, fft_poisson

function fft_grid_kk(N, eps = 1e-6)
    hx = 2π / (N + 1)
    xx = Array{Float64}(undef, N + 1)
    
    # xx
    N_2 = floor(Int, 0.5 * N)
    for i in 1:N_2
        xx[i +  1] = hx * i
    end
    
    if isodd(N)
        N_2 = floor(Int, 0.5*(N + 1))
        for i in 1:N_2
            xx[i + N_2] = hx * (i - N_2 - 1)
        end
    else
        N_2 = floor(Int, 0.5*(N + 1))
        for i in 1:N_2
            xx[i + N_2 + 1] = hx * (i - N_2 - 1)
        end
    end
    
    xx[1] = eps # make sure that the denominator is not zero
    return xx
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,1}, boundary::Periodic, Device::CPU) where T
    rho_bar = fft(rho)    
    rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for i in 1:Len[1]+1
        u_bar[i] = rho_bar[i] / (delta2sum + delta2[1] * cos(xx[i]))
    end
    
    u = real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,1}, boundary::Periodic, Device::GPU) where T
    rho_bar = fft(rho)    
    CUDA.@allowscalar rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    
    u_bar = rho_bar ./ (delta2sum .+ delta2[1] * cos.(CuArray(xx * ones(Len[2]+1)')))
    
    u = real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,2}, boundary::Periodic, Device::CPU) where T
    rho_bar = fft(rho)
    rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    yy = fft_grid_kk(Len[2])
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for j in 1:Len[2]+1
        for i in 1:Len[1]+1
            u_bar[i,j] = rho_bar[i,j] / (delta2sum + delta2[1] * cos(xx[i]) + delta2[2] * cos(yy[j]))
        end
    end
    
    u = real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,2}, boundary::Periodic, Device::GPU) where T
    rho_bar = fft(rho)
    CUDA.@allowscalar rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    yy = fft_grid_kk(Len[2])
    
    # solve u_bar
    u_bar = rho_bar ./ (delta2sum .+ delta2[1] * cos.(CuArray(xx * ones(Len[2]+1)')) + delta2[2] * cos.(CuArray(ones(Len[1]+1) * yy')))
    
    u = real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,3}, boundary::Periodic, Device::CPU) where T
    rho_bar = fft(rho)    
    rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    yy = fft_grid_kk(Len[2])
    zz = fft_grid_kk(Len[3])
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for k in 1:Len[3]+1
        for j in 1:Len[2]+1
            for i in 1:Len[1]+1
                u_bar[i,j,k] = rho_bar[i,j,k] / (delta2sum + delta2[1] * cos(xx[i]) + delta2[2] * cos(yy[j]) + delta2[3] * cos(zz[k]))
            end
        end
    end
    
    u = real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,3}, boundary::Periodic, Device::GPU) where T
    rho_bar = fft(rho)    
    CUDA.@allowscalar rho_bar[1] *= 0.0
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    xx = fft_grid_kk(Len[1])
    yy = fft_grid_kk(Len[2])
    zz = fft_grid_kk(Len[3])
    
    oneMatrix = cu(ones((Len.+1)...))
    dcx = delta2[1] .* cos.(oneMatrix .* cu(xx))
    dcy = delta2[2] .* cos.(oneMatrix .* cu(yy'))
    dcz = delta2[3] .* cos.(oneMatrix .* cu(reshape(zz, 1, 1, Len[3]+1)))
    u_bar = rho_bar ./ (delta2sum .+ dcx .+ dcy .+ dcz)
    CUDA.@allowscalar u_bar[1] = 0.0f0+0.0f0*im
    
    u = real(ifft(u_bar))
end

### Homogeneous Dirichlet boundary conditions - fast sine transform
function fft_poisson(Δ, Len, rho::AbstractArray{T,1}, boundary::Dirichlet, Device::CPU) where T
    #rho_bar = fft(mesh.rho)
    #rho_bar[1] *= 0.0
    rho_bar = FFTW.r2r(complex(rho[2:end-1]), FFTW.RODFT00)
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    hx = pi / (Len[1] + 1)
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for i in 1:Len[1]-1
        u_bar[i] = rho_bar[i] / (delta2sum + delta2[1] * cos(hx * i))
    end

    u = real(FFTW.r2r(u_bar, FFTW.RODFT00)/((2*(Len[1] + 1))))
    #mesh.phi .= real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,2}, boundary::Dirichlet, Device::CPU) where T
    #rho_bar = fft(mesh.rho)
    #rho_bar[1] *= 0.0
    rho_bar = FFTW.r2r(complex(rho[2:end-1, 2:end-1]), FFTW.RODFT00)
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    hx = pi / (Len[1] + 1)
    hy = pi / (Len[2] + 1)
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for j in 1:Len[2]-1
        for i in 1:Len[1]-1
            u_bar[i,j] = rho_bar[i,j] / (delta2sum + delta2[1] * cos(hx * i) + delta2[2] * cos(hy * j))
        end
    end

    u = real(FFTW.r2r(u_bar, FFTW.RODFT00)/((2*(Len[1] + 1)) * (2*(Len[2] + 1))))
    #mesh.phi .= real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,3}, boundary::Dirichlet, Device::CPU) where T
    #rho_bar = fft(mesh.rho)
    #rho_bar[1] *= 0.0
    rho_bar = FFTW.r2r(complex(rho[2:end-1, 2:end-1, 2:end-1]), FFTW.RODFT00)
    
    delta2 = 2 ./ (ustrip.(Δ) .^ 2)
    delta2sum = - sum(delta2)
    
    hx = pi / (Len[1] + 1)
    hy = pi / (Len[2] + 1)
    hz = pi / (Len[3] + 1)
    
    # solve u_bar
    u_bar = similar(rho_bar);
    for k in 1:Len[3]-1
        for j in 1:Len[2]-1
            for i in 1:Len[1]-1
                u_bar[i,j,k] = rho_bar[i,j,k] / (delta2sum + delta2[1] * cos(hx * i) + delta2[2] * cos(hy * j) + delta2[3] * cos(hz * k))
            end
        end
    end

    u = real(FFTW.r2r(u_bar, FFTW.RODFT00)/((2*(Len[1] + 1)) * (2*(Len[2] + 1)) * (2*(Len[3] + 1))))
    #mesh.phi .= real(ifft(u_bar))
end

function fft_poisson(Δ, Len, rho::AbstractArray{T,3}, pos, boundary::Vacuum, Device::CPU) where T
    # Step.1 Zero Dirichlet boundary
    phi = rho .* 0
    phi[2:end-1,2:end-1,2:end-1] .= fft_poisson(Δ, Len, rho, Dirichlet(), Device) * 4π

    Nx, Ny, Nz = size(phi)
    cell_volumn = prod(Δ)

    # Step.2 Surfac screening charges. Right-hand coordinates
    charge_nth = -phi[   Nx-1, 2:end-1, 2:end-1] / (4π * Δ[1]^2) * cell_volumn # Δ[2] * Δ[3] # x+
    charge_sth = -phi[      2, 2:end-1, 2:end-1] / (4π * Δ[1]^2) * cell_volumn # Δ[2] * Δ[3] # x-
    charge_wst = -phi[2:end-1,    Ny-1, 2:end-1] / (4π * Δ[2]^2) * cell_volumn # Δ[1] * Δ[3] # y+
    charge_est = -phi[2:end-1,       2, 2:end-1] / (4π * Δ[2]^2) * cell_volumn # Δ[1] * Δ[3] # y-
    charge_top = -phi[2:end-1, 2:end-1,    Nz-1] / (4π * Δ[3]^2) * cell_volumn # Δ[1] * Δ[2] # z+
    charge_bot = -phi[2:end-1, 2:end-1,       2] / (4π * Δ[3]^2) * cell_volumn # Δ[1] * Δ[2] # z-

    # Step.3 Direct-summation to compute all-space potentials from surface charges
    @info "Computing screening potential: nth & sth"
    phi_screening = 0 * phi
    for k in 2:Nz-1
        for j in 2:Ny-1
            # nth
            charge_index = CartesianIndex(Nx,j,k)
            charge_pos = pos[charge_index]
            charge_phi = charge_nth[j-1, k-1] ./ norm.(charge_pos .- pos) # the potentials do not have (-) sign, because the screen charges have the opposite sign
            charge_phi[charge_index] = 0
            phi_screening += charge_phi

            # sth
            charge_index = CartesianIndex(1,j,k)
            charge_pos = pos[charge_index]
            charge_phi = charge_sth[j-1, k-1] ./ norm.(charge_pos .- pos)
            charge_phi[charge_index] = 0
            phi_screening += charge_phi
        end
    end

    @info "Computing screening potential: wst & est"
    for k in 2:Nz-1
        for i in 2:Nx-1
            # wst
            charge_index = CartesianIndex(i,Ny,k)
            charge_pos = pos[charge_index]
            charge_phi = charge_wst[i-1, k-1] ./ norm.(charge_pos .- pos)
            charge_phi[charge_index] = 0
            phi_screening += charge_phi

            # est
            charge_index = CartesianIndex(i,1,k)
            charge_pos = pos[charge_index]
            charge_phi = charge_est[i-1, k-1] ./ norm.(charge_pos .- pos)
            charge_phi[charge_index] = 0
            phi_screening += charge_phi
        end
    end

    @info "Computing screening potential: top & bot"
    for j in 2:Nz-1
        for i in 2:Nx-1
            # top
            charge_index = CartesianIndex(i,j,Nz)
            charge_pos = pos[charge_index]
            charge_phi = charge_top[i-1, j-1] ./ norm.(charge_pos .- pos)
            charge_phi[charge_index] = 0
            phi_screening += charge_phi

            # bot
            charge_index = CartesianIndex(i,j,1)
            charge_pos = pos[charge_index]
            charge_phi = charge_bot[i-1, j-1] ./ norm.(charge_pos .- pos)
            charge_phi[charge_index] = 0
            phi_screening += charge_phi
        end
    end

    ## double transforms


    # return phi_screening

    # Step.4 Simply subtract the screening potential
    phi -= phi_screening

    return phi
end


# Dirichlet BC returns a smaller array
function fft_poisson!(m::MeshCartesianStatic, rho::AbstractArray, boundary::Periodic)
    m.phi .= fft_poisson(m.config.Δ, m.config.Len, rho, boundary, m.config.device) .* unit(eltype(m.phi)) * 4π
end

function fft_poisson!(m::MeshCartesianStatic, rho::AbstractArray{T,1}, boundary::Dirichlet) where T
    m.phi[2:end-1] .= fft_poisson(m.config.Δ, m.config.Len, rho, boundary, m.config.device) .* unit(eltype(m.phi)) * 4π # TODO: c_D = ?
end

function fft_poisson!(m::MeshCartesianStatic, rho::AbstractArray{T,2}, boundary::Dirichlet) where T
    m.phi[2:end-1,2:end-1] .= fft_poisson(m.config.Δ, m.config.Len, rho, boundary, m.config.device) .* unit(eltype(m.phi)) * 4π # TODO: c_D = 2π ?
end

function fft_poisson!(m::MeshCartesianStatic, rho::AbstractArray{T,3}, boundary::Dirichlet) where T
    m.phi[2:end-1,2:end-1,2:end-1] .= fft_poisson(m.config.Δ, m.config.Len, rho, boundary, m.config.device) .* unit(eltype(m.phi)) * 4π
    # m.phi .= fft_poisson(m.config.Δ, m.config.Len, rho, boundary, m.config.device) .* unit(eltype(m.phi))
end

function fft_poisson!(m::MeshCartesianStatic, rho::AbstractArray{T,3}, boundary::Vacuum) where T
    m.phi .= fft_poisson(m.config.Δ, m.config.Len, rho, m.pos, boundary, m.config.device) .* unit(eltype(m.phi))
end

function fft_poisson(m::AbstractMesh, G::Number)
    fft_poisson!(m, ustrip.(m.rho .* G), m.config.boundary)
end

include("precompile.jl")

end # module PhysicalFFT
