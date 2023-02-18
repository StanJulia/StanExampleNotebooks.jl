#### README for the Pluto notebooks in StanExampleNotebooks.jl.

All example notebooks require a working `cmdstan` installation. See [here](https://github.com/StanJulia/StanSample.jl/blob/master/README.md).

## Usage

To (locally) reproduce and use this project, do the following:

1. Download this [project](https://github.com/StanJulia/StanExampleNotebooks.jl) from Github and move to the downloaded directory, e.g.:

```
$ cd .julia/dev
$ git clone https://github.com/StanJulia/StanExampleNotebooks.jl StanExampleNotebooks
$ cd StanExampleNotebooks
```

The next step assumes your `basic` Julia environment includes `Pkg` and `Pluto`.

2. Start a Pluto notebook server.
```
$ cd notebooks
$ julia

julia> using Pluto
julia> Pluto.run()
```

A Pluto page should open in a browser. See [this page](https://www.juliafordatascience.com/first-steps-5-pluto/) for a quick Pluto introduction.

3. Select a notebook in the `open a file` entry box, e.g. type `./` and select a notebook. 

