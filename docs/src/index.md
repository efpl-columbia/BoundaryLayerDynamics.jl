# Turbulence-Resolving Simulations of Boundary-Layer Flows

```@meta
CurrentModule = BoundaryLayerDynamics
```

BoundaryLayerDynamics.jl can simulate a range of physical problems, but its primary purpose is to simulate the evolution of turbulent flows in simplified representations of the atmospheric boundary-layer.

## Main Characteristics

- Direct numerical simulation (DNS) and large-eddy simulation (LES) of incompressible flow dynamics.
- Three-dimensional Cartesian domain, with periodic boundary conditions along ``x₁`` and ``x₂``.
- Spatial discretization based on truncated Fourier expansion along ``x₁`` and ``x₂``.
- Spatial discretization based on second-order (central) finite differences on a staggered grid along ``x₃``.
- Explicit integration in time.
- CPU-based computation with distributed-memory parallelization along ``x₃`` using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface).

## Usage Overview

Running simulation consists of setting up a **[`Model`](@ref)** and then simulating its dynamics with **[`evolve!`](@ref)**, collecting output data along the way.

The **[Setup & Workflow](@ref)** page gives practical advice on setting up a project with BoundaryLayerDynamics.jl and running simulations both on personal machines and high-performance computing systems.

The **[Computational Model](@ref)** page explains how to set up a [`Model`](@ref) by configuring the desired [computational domain](@ref Domain), resolution of the [discretization](@ref Discretization), and [physical processes](@ref Physical-Processes).
It also documents the mathematical concepts that the computational model is based on.

The **[Evolution in Time](@ref)** page explains how to run a simulation using [`evolve!`](@ref), after configuring the [time-integration method](@ref Time-Integration-Methods) and the [output modules](@ref Output-Modules).
It also documents how the time integration handles processes that are assumed to act instantaneously such as the pressure in incompressible flows.
