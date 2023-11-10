# To make it easier to build the documentation, we first set up the package
# environment such that both Documenter and the current version of
# BoundaryLayerDynamics can be loaded.
using Pkg
# (0) skip setup when run repeatedly by LiveServer
if !startswith(Pkg.project().path, tempdir())
    # (1) instantiate the BoundaryLayerDynamics.jl environment
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    # (2) set up a temporary environment that includes Documenter
    Pkg.activate(; temp=true)
    Pkg.add("Documenter")
    # (3) make sure the load path only includes these two environments
    #     and the standard library
    empty!(LOAD_PATH)
    push!(LOAD_PATH, "@", joinpath(@__DIR__, ".."), "@stdlib")
end

using Documenter, BoundaryLayerDynamics

makedocs(sitename = "BoundaryLayerDynamics.jl",
         pages = ["Introduction" => "index.md",
                  "workflow.md", "model.md", "time-evolution.md",
                  "Physical Processes" => [
                    "processes/diffusion.md",
                    "processes/momentum-advection.md",
                    "processes/pressure.md",
                    "processes/sources.md",
                    "processes/sgs-advection.md",
                   ],
                  "wall-models.md",
                 ],
         modules = [BoundaryLayerDynamics],
         repo = Remotes.GitHub("efpl-columbia", "BoundaryLayerDynamics.jl"),
         checkdocs = :none, # consider using :exports instead
         format = Documenter.HTML(edit_link = nothing,
                                  prettyurls = get(ENV, "PRETTY_URLS", "") == "true",
                                  sidebar_sitename = false),
        )
