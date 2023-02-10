#### README for the Pluto notebooks in StanNotebookExamples.jl.

Running these notebooks requires a Julia installation with only the packages `Pkg` and `Pluto` installed.
The example scripts requires a working `cmdstan` installation. See [here](https://github.com/StanJulia/StanSample.jl/blob/master/README.md).

Change to this directory and start Pluto:
```
cd "this directory"
julia

julia> using Pluto
julia> Pluto.run()   # A page should open up in your default browser
```

Select a notebook in Pluto "Open a notebook" box, e.g. type "./".

See [this](https://github.com/fonsp/Pluto.jl/) page for more details.
