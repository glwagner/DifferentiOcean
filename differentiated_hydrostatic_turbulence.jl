using Oceananigans
using Random
using Enzyme

Random.seed!(123)
arch = CPU()
Nx = Ny = 32
halo = (3, 3, 3)
x = y = (0, 2π)
z = (0, 1)
g = 4^2
c = sqrt(g)

grid = RectilinearGrid(arch, size=(Nx, Ny, 3); x, y, z, halo, topology=(Periodic, Periodic, Bounded))
closure = ScalarDiffusivity(ν=1e-2)
momentum_advection = Centered(order=2)
free_surface = ExplicitFreeSurface(gravitational_acceleration=g)
model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, free_surface, closure)

ϵ(x, y, z) = 2randn() - 1
set!(model, u=ϵ, v=ϵ)

Δx = minimum_xspacing(grid)
Δt = 0.01 * Δx / c
for n = 1:100
    time_step!(model, Δt)
end

u_init = deepcopy(model.velocities.u)
v_init = deepcopy(model.velocities.v)

for n = 1:10
    time_step!(model, Δt)
end

u_truth = deepcopy(model.velocities.u)
v_truth = deepcopy(model.velocities.v)

function viscous_hydrostatic_turbulence(ν, model, u_init, v_init, Δt, u_truth, v_truth)
    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    new_closure = ScalarDiffusivity(; ν, κ=NamedTuple())
    model.closure = new_closure
    set!(model, u=u_init, v=v_init)

    # Step it forward
    for n = 1:10
        time_step!(model, Δt)
    end

    # Compute the sum square error
    u, v, w = model.velocities
    Nx, Ny, Nz = size(model.grid)
    err = 0.0
    for j = 1:Ny, i = 1:Nx 
        err += @inbounds (u[i, j, 3] - u_truth[i, j, 3])^2 + 
                         (v[i, j, 3] - v_truth[i, j, 3])^2
    end

    return err::Float64
end

# Use a manual finite difference to compute a gradient
Δν = 1e-6
ν1 = 2e-2
ν2 = ν1 + Δν
e1 = viscous_hydrostatic_turbulence(ν1, model, u_init, v_init, Δt, u_truth, v_truth)
e2 = viscous_hydrostatic_turbulence(ν2, model, u_init, v_init, Δt, u_truth, v_truth)
ΔeΔν = (e2 - e1) / Δν

@info "Finite difference computed: $ΔeΔν"

@info "Now with autodiff..."
start_time = time_ns()

# Use autodiff to compute a gradient
dmodel = Enzyme.make_zero(model)
dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
                viscous_hydrostatic_turbulence,
                Active(ν1),
                Duplicated(model, dmodel),
                Const(u_init),
                Const(v_init),
                Const(Δt),
                Const(u_truth),
                Const(v_truth))

@info "Automatically computed: $dedν."
@info "Elapsed time: " * prettytime(1e-9 * (time_ns() - start_time))

