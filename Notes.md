
### Tag version notes

1. git commit -m "Tag v0.5.1: changes"
2. git tag v0.5.1
3. git push origin main --tags

### Cloning the repository

```
# Cd to where you would like to clone to
$ git clone https://github.com/StanJulia/StanExampleNotebooks.jl StanExampleNotebooks
$ cd StanExampleNotebooks/notebooks
$ julia
```
and in the Julia REPL:

```
julia> using Pluto
julia> Pluto.run()
julia>
```

Pluto opens a notebook in your default browser.

### Extract .jl from Jupyter notebook (`jupytext` needs to be installed)

# jupytext --to jl "./ch7.ipynb"
