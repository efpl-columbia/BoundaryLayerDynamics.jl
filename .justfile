documenter_version := "1.1.2"
liveserver_version := "1.2.7"

_default:
  @just --list

# Run automated tests
test *ARGS:
  #!/usr/bin/env julia
  import Pkg
  Pkg.activate(".")
  Pkg.test(; test_args = string.(split("{{ARGS}}")))

# Build documentation to `docs/build` folder
makedocs:
  #!/usr/bin/env julia
  import Pkg
  Pkg.activate(; temp=true)
  empty!(LOAD_PATH)
  push!(LOAD_PATH, "@", "@stdlib")
  Pkg.develop(path = ".")
  Pkg.add(Pkg.PackageSpec(name="Documenter", version="{{documenter_version}}"))
  include(joinpath(pwd(), "docs", "make.jl"))

# Launch a local server for the documentation
servedocs:
  #!/usr/bin/env julia
  import Pkg
  Pkg.activate(; temp=true)
  empty!(LOAD_PATH)
  push!(LOAD_PATH, "@", "@stdlib")
  Pkg.develop(path = ".")
  Pkg.add(Pkg.PackageSpec(name="Documenter", version="{{documenter_version}}"))
  Pkg.add(Pkg.PackageSpec(name="LiveServer", version="{{liveserver_version}}"))
  import LiveServer
  LiveServer.servedocs()
