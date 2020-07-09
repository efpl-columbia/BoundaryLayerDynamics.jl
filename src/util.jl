equivalently(args...) = all(arg === args[1] for arg=args[2:end]) ?
                        args[1] : error("Arguments are not equivalent")

reset!(a::AbstractArray{T}) where T = (a .= zero(T); a)

# currently unused, could be removed
printdirect(s...) = MPI.Initialized() ? println("process ", MPI.Comm_rank(MPI.COMM_WORLD) + 1, " ", s...) : println(s...)
