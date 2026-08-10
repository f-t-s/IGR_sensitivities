module IGRAdjoints1D

using LinearAlgebra
using FastGaussQuadrature
using Enzyme

include("basis.jl")
include("mesh.jl")
include("euler.jl")
include("hyperbolic.jl")
include("elliptic.jl")
include("initial_conditions.jl")
include("forward.jl")
include("sensitivity.jl")
include("enzyme_adjoint.jl")
include("adjoint_pde.jl")

end # module
