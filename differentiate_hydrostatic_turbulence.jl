using Oceananigans
using Enzyme

arch = CPU()
Nx = Ny = 64
halo = (3, 3, 3)
x = y = (0, 2π)
z = (0, 1)
g = 4^2
c = sqrt(g)

grid = RectilinearGrid(arch, size=(Nx, Ny, 3); x, y, z, halo, topology=(Periodic, Periodic, Bounded))
closure = ScalarDiffusivity(ν=1e-3)
momentum_advection = WENO(order=5)
free_surface = ExplicitFreeSurface(gravitational_acceleration=4^2)
model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, free_surface, closure)

ϵ(x, y, z) = 2randn() - 1
set!(model, u=ϵ, v=ϵ)

u_init = Array(interior(model.velocities.u, :, :, 3))
v_init = Array(interior(model.velocities.v, :, :, 3))

Δx = minimum_xspacing(grid)
Δt = 0.05 * Δx / c
for n = 1:10
    time_step!(model, Δt)
end

u_truth = Array(interior(model.velocities.u, :, :, 3))
v_truth = Array(interior(model.velocities.v, :, :, 3))

function viscous_hydrostatic_turbulence(ν, model, ui, vi, Δt, u_truth, v_truth)
    model.clock.iteration = 0
    model.clock.time = 0

    new_closure = ScalarDiffusivity(; ν, κ=NamedTuple())
    model.closure = new_closure
    set!(model, u=ui, v=vi)
    for n = 1:10
        time_step!(model, Δt)
    end

    u_model = Array(interior(model.velocities.u, :, :, 3))
    v_model = Array(interior(model.velocities.v, :, :, 3))

    δu² = @. (u_model - u_truth)^2 
    δv² = @. (v_model - v_truth)^2 
    err = sum(δu²)

    return err::Float64
end

ν1 = 2e-3
ν2 = 2.1e-3
e1 = viscous_hydrostatic_turbulence(ν1, model, u_init, v_init, Δt, u_truth, v_truth)
e2 = viscous_hydrostatic_turbulence(ν2, model, u_init, v_init, Δt, u_truth, v_truth)

@show (e2 - e1) / (ν2 - ν1)

# Autodiff
dmodel = Enzyme.make_zero(model)
ν0 = 2e-3

dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
                viscous_hydrostatic_turbulence,
                Active(ν0),
                Duplicated(model, dmodel),
                Const(u_init),
                Const(v_init),
                Const(Δt),
                Const(u_truth),
                Const(v_truth))

