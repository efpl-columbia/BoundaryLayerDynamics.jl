module Helpers

export equivalently, reset!, sequentially

using MPI: MPI

equivalently(args...) = all(arg === args[1] for arg=args[2:end]) ?
                        args[1] : error("Arguments are not equivalent")

function sequentially(f, comm)
    MPI.Initialized() || return f()
    s, r = MPI.Comm_size(comm), MPI.Comm_rank(comm) + 1
    for i=1:s
        # TODO: propagate return value of f()
        i == r && f()
        MPI.Barrier(comm)
    end
end

function approxdiv(a,b)
    result_float = a/b
    result_int = round(Int, result_float)
    isapprox(result_float, result_int) || error("Division has a non-integer result")
    result_int
end

# currently unused, could be removed
printdirect(s...) = MPI.Initialized() ? println("process ", MPI.Comm_rank(MPI.COMM_WORLD) + 1, " ", s...) : println(s...)

reset!(fields::NamedTuple) = (fill!.(values(fields), 0); fields)

end # module Helpers
