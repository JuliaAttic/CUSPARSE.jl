using Documenter, CUSPARSE

makedocs(
         modules=[CUSPARSE]
        )

deploydocs(
    repo = "github.com/JuliaGPU/CUSPARSE.jl.git",
    branch = "gh-pages"
    )

