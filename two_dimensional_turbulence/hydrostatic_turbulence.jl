using Oceananigans

arch = CPU()
Nx = Ny = 64
halo = (3, 3, 3)
x = y = (0, 2π)
z = (0, 1)
g = 4^2
c = sqrt(g)

grid = RectilinearGrid(arch, size=(Nx, Ny, 3); x, y, z, halo, topology=(Periodic, Periodic, Bounded))

momentum_advection = WENO(order=5)
free_surface = ExplicitFreeSurface(gravitational_acceleration=g)
model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, free_surface)

ϵ(x, y, z) = 2randn() - 1
set!(model, u=ϵ, v=ϵ)

Δx = minimum_xspacing(grid)
Δt = 0.05 * Δx / c
simulation = Simulation(model; Δt, stop_iteration=1000)

progress(sim) = @info string(iteration(sim), ", time: ", time(sim))
add_callback!(simulation, progress, IterationInterval(10))

run!(simulation)

using GLMakie

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)

fig = Figure(size=(900, 400))
axz = Axis(fig[1, 1], aspect=1)
axw = Axis(fig[1, 2], aspect=1)
heatmap!(axz, view(ζ, :, :, grid.Nz), colormap=:balance)
heatmap!(axw, view(w, :, :, grid.Nz), colormap=:balance)
display(current_figure())

