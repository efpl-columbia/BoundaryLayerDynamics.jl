function runbmk(np, fn = nothing)
    bmkdir = abspath(".")
    io = PipeBuffer()
    cmd = `mpirun --quiet -np $np julia --project=$bmkdir "$bmkdir/suite.jl"`
    run(pipeline(cmd, stdout=io))
    data = read(io, String)
    isnothing(fn) || write(fn, data)
    data
end

for np = (32, 16, 8, 4,)
    # this should take O(10min) for np>1 on leoncina
    runbmk(np, "./bmk-$np.json")
end
