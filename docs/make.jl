using Documenter, ChannelFlow

makedocs(sitename = "ChannelFlow.jl",
         pages = ["Overview" => "index.md", "installation.md", "usage.md"],
         format = Documenter.HTML(edit_link = nothing),
        )
