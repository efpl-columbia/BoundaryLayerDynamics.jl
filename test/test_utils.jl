function setup_random_fields(T, n1, n2, n3)
    gd = BLD.Grid((n1, n2, n3))
    uh = BLD.zeros(T, gd, NS(:C))
    uv = BLD.zeros(T, gd, NS(:I))
    fill!(uh, rand(Complex{T}))
    fill!(uv, rand(Complex{T}))
    ht = BLD.PhysicalSpace.Transform2D(T, BLD.PhysicalSpace.default_size(gd))
    gd, ht, uh, uv
end

function test_convergence(
    N,
    ε;
    exponential = false,
    order = nothing,
    threshold_linearity = 0.99,
    threshold_slope = 0.95,
    verbose = false,
)
    X = exponential ? N : log.(N)
    Y = log.(ε)

    cov(x, y) = (n = length(x);
    n == length(y) || error("Length does not match");
    μx = sum(x) / n;
    μy = sum(y) / n;
    sum((x .- μx) .* (y .- μy)) ./ (n - 1))

    varX = cov(X, X)
    varY = cov(Y, Y)
    covXY = cov(X, Y)
    slope = covXY / varX
    Rsq = covXY^2 / (varX * varY)

    @test Rsq > threshold_linearity

    if order != nothing
        @test -slope > order * threshold_slope
        verbose && println("Passed with order $(-slope) (R²=$(Rsq))")
    end

    -slope, Rsq
end

# These MPI test tools provide useful functionality to test MPI programs. They
# are independent from this project and could be moved to a separate package in
# the future, once they have matured and appear useful for other software too.

import Test

# the first set of tools is used by the “host”, i.e. the regular, serial
# runtests file, to launch tests running inside MPI processes

struct MPITestSetException <: Exception
    nprocs::Int
    caller::String
    MPITestSetException(np, c = "") = new(np, c)
end

function Base.showerror(io::IO, e::MPITestSetException)
    printstyled(io, "Error During MPI Tests (n=", e.nprocs, ")"; bold = true, color = Base.error_color())
    if length(e.caller) > 0
        print(io, " at ")
        printstyled(io, e.caller; bold = true)
    end
end

function run_mpi_test(file, nprocs::Integer, subset = [])
    juliabin = joinpath(Sys.BINDIR, Base.julia_exename())
    project = Base.active_project()
    path = joinpath(dirname(@__FILE__), file)
    try
        run(`$(MPI.mpiexec()) -n $(nprocs) $(juliabin) --color=yes "--project=$(project)" $(path) --mpi $subset`)
    catch
        st = stacktrace()[2:end]
        showerror(stderr, MPITestSetException(nprocs, string(st[1].file, ":", st[1].line)), st)
        println()
        exit(1)
    end
end

# the second set of tools is used by the test script running in parallel on a
# number of MPI ranks, in order to collect test results and avoid printing
# errors multiple times from different ranks

printall(vars...) = if MPI.Initialized()
        for i in 1:MPI.Comm_size(MPI.COMM_WORLD)
            i == MPI.Comm_rank(MPI.COMM_WORLD) + 1 && println("process ", i, ": ", vars...)
            MPI.Barrier(MPI.COMM_WORLD)
        end
    else
        println(vars...)
    end

show_all(var) = for i in 1:MPI.Comm_size(MPI.COMM_WORLD)
        i == MPI.Comm_rank(MPI.COMM_WORLD) + 1 && show(var)
        MPI.Barrier(MPI.COMM_WORLD)
    end

# specifying T avoids accidentially taking the minimum in Fourier space
global_minimum(val::T) where {T<:Real} = MPI.Initialized() ? MPI.Allreduce(val, MPI.MIN, MPI.COMM_WORLD) : val
global_minimum(field::Array{T}) where {T<:Real} = global_minimum(mapreduce(abs, min, field))

# specifying T avoids accidentially taking the maximum in Fourier space
global_maximum(val::T) where {T<:Real} = MPI.Initialized() ? MPI.Allreduce(val, MPI.MAX, MPI.COMM_WORLD) : val
global_maximum(field::Array{T}) where {T<:Real} = global_maximum(mapreduce(abs, max, field))

global_sum(x) = MPI.Initialized() ? MPI.Allreduce(sum(x), +, MPI.COMM_WORLD) : sum(x)

function global_vector(x)
    x = collect(x)
    MPI.Initialized() || return x
    l = convert(Vector{Cint}, MPI.Allgather(length(x), MPI.COMM_WORLD))
    MPI.Allgatherv(x, l, MPI.COMM_WORLD)
end

function mktempdir_parallel(f)
    MPI.Initialized() || return mktempdir(f)
    mktempdir_once(f0) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? mktempdir(f0) : f0("")
    mktempdir_once() do p
        p = MPI.bcast(p, 0, MPI.COMM_WORLD)
        rval = f(p)
        MPI.Barrier(MPI.COMM_WORLD)
        rval
    end
end

struct MPITestSet <: Test.AbstractTestSet
    description::AbstractString
    results::Vector
    fail_ranks::Vector{Int}
    error_ranks::Vector{Int}
    mpi_rank::Integer
    mpi_size::Integer
    MPITestSet(desc, initialized) = new(desc, [], [], [], MPI.Comm_rank(MPI.COMM_WORLD), MPI.Comm_size(MPI.COMM_WORLD))
end

# constructor takes a description string and options keyword arguments
MPITestSet(desc; kwargs...) = MPI.Initialized() ? MPITestSet(desc, true) : Test.DefaultTestSet(desc; kwargs...)

list_ranks(r) =
    length(r) == 1 ? string("rank ", r[1], " only") : string("ranks ", join(r[1:end-1], ", "), " & ", r[end])

function Test.record(ts::MPITestSet, child::Test.AbstractTestSet)
    push!(ts.fail_ranks, sum(child.fail_ranks .!= 0))
    push!(ts.error_ranks, sum(child.error_ranks .!= 0))
    push!(ts.results, child)
end

function Test.record(ts::MPITestSet, t::Test.Result)

    # print error right away, in case only some processes encounter it
    if t isa Test.Error
        # TODO: do not simply skip errors
        println("[Rank ", ts.mpi_rank, "] ", t)
    else
        # this code still has some references to errors, since we originally
        # had coordinated reporting errors between all processes. this leads
        # to deadlocks without any output when some processes encounter errors
        # while other processes continue and wait for MPI communication from
        # others.
        # TODO: find a clean version of handling errors in- & outside of tests
        errid = Int8(t isa Test.Error ? 2 : t isa Test.Fail ? 1 : 0)
        errids = MPI.Allgather(errid, MPI.COMM_WORLD)
        if ts.mpi_rank + 1 == findfirst(!iszero, errids) # only report one error
            fails = findall(errids .== 1) .- 1
            errs = findall(errids .== 2) .- 1
            print("Rank ", ts.mpi_rank, ": ")
            print(t)
            summary = string(
                "\n Test",
                length(fails) > 0 ? " failed on " * list_ranks(fails) : "",
                length(fails) > 0 && length(errs) > 0 ? " and" : "",
                length(errs) > 0 ? " produced errors on " * list_ranks(errs) : "",
                "\n",
            )
            printstyled(summary; color = Base.error_color())
        end
        push!(ts.fail_ranks, sum(errids .== 1))
        push!(ts.error_ranks, sum(errids .== 2))
        push!(ts.results, t)
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

function testresults(ts::MPITestSet)
    total, passed = 0, 0
    for (i, r) in enumerate(ts.results)
        if r isa Test.Result
            total += 1
            passed += (ts.fail_ranks[i] == ts.error_ranks[i] == 0 ? 1 : 0)
        else
            t, p, = testresults(r)
            total += t
            passed += p
        end
    end
    total, passed
end

function print_test_summary(ts::MPITestSet, prefix = "")
    total, passed = testresults(ts)
    print(ts.description, " (")
    printstyled("$passed/$total passed"; color = (total == passed ? :green : :red))
    println(")")
    total == passed && return
    for (i, r) in enumerate(ts.results)
        if r isa Test.Result
            f = ts.fail_ranks[i]
            e = ts.error_ranks[i]
            if f > 0 || e > 0
                println(prefix, "– Test ", i, ": ", ts.mpi_size - f - e, " Pass", ", ", f, " Fail", ", ", e, " Err")
            end
        elseif r isa MPITestSet
            print(prefix, "• ")
            print_test_summary(r, prefix * "  ")
        end
    end
end

function Test.finish(ts::MPITestSet)

    # just record if we're not the top-level parent
    if Test.get_testset_depth() > 0
        Test.record(Test.get_testset(), ts)
        return ts
    end

    MPI.Barrier(MPI.COMM_WORLD)
    if ts.mpi_rank == 0
        printstyled("Test Summary:"; bold = true, color = :white)
        println(" (", ts.mpi_size, " MPI process$(ts.mpi_size > 1 ? "es" : ""))")
        print_test_summary(ts)
    end

    if sum(ts.fail_ranks) + sum(ts.error_ranks) > 0
        MPI.Finalize()
        exit(1)
    end

    return ts
end
