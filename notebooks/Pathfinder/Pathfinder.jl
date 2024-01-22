### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 5084b8f0-65ac-4704-b1fc-2a9008132bd7
using Pkg

# ╔═╡ d7753cf6-7452-421a-a3ec-76e07646f808
Pkg.activate(expanduser("~/.julia/dev/StanPathfinder"))

# ╔═╡ 550371ad-d411-4e66-9d63-7329322c6ea1
begin
    # Specific to this notebook
    using StanPathfinder
    using DataFrames
    using StanIO: read_csvfiles
end

# ╔═╡ eb7ea04a-da52-4e69-ac3e-87dc7f014652
md"## Pathfinder example"

# ╔═╡ cf39df58-3371-4535-88e4-f3f6c0404500
md" ###### Widen the cells."

# ╔═╡ 0616ece8-ccf8-4281-bfed-9c1192edf88e
html"""
<style>
    main {
        margin: 0 auto;
        max-width: 2000px;
        padding-left: max(160px, 10%);
        padding-right: max(160px, 15%);
    }
</style>
"""

# ╔═╡ 4755dab0-d228-41d3-934a-56f2863a5652
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ da214635-a5ff-4c64-8930-683b11bd8531
bernoulli_model = "
data { 
  int<lower=1> N; 
  array[N] int<lower=0,upper=1> y;
} 
parameters {
  real<lower=0,upper=1> theta;
} 
model {
  theta ~ beta(1,1);
  y ~ bernoulli(theta);
}
";

# ╔═╡ 104b926e-5a36-41d7-a587-aeafa7bcd9c5
data = Dict("N" => 10, "y" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]);

# ╔═╡ eb4a78c4-0522-44ae-b3ab-fe489c135026
md" 
!!! note

Keep tmpdir across multiple runs to prevent re-compilation."

# ╔═╡ 1ae7dd74-40c5-439c-a95c-3d31eecc8d89
tmpdir = joinpath(@__DIR__, "tmp")

# ╔═╡ a2939ec8-3ac1-4a7b-8e0b-68b15f0d98ce
begin
	sm = PathfinderModel("bernoulli", bernoulli_model, tmpdir)
	rc = stan_pathfinder(sm; data, seed=rand(1:200000000, 1)[1], num_paths=6, num_threads=1)
end;

# ╔═╡ dd2ef429-7294-4602-895e-73031b3e8314
if success(rc)
	df = read_csvfiles(sm.file, :dataframe)
end

# ╔═╡ aa891570-ad41-43d2-81c9-dd4303343f71
create_pathfinder_profile_df(sm)

# ╔═╡ Cell order:
# ╟─eb7ea04a-da52-4e69-ac3e-87dc7f014652
# ╟─cf39df58-3371-4535-88e4-f3f6c0404500
# ╠═0616ece8-ccf8-4281-bfed-9c1192edf88e
# ╟─4755dab0-d228-41d3-934a-56f2863a5652
# ╠═5084b8f0-65ac-4704-b1fc-2a9008132bd7
# ╠═d7753cf6-7452-421a-a3ec-76e07646f808
# ╠═550371ad-d411-4e66-9d63-7329322c6ea1
# ╠═da214635-a5ff-4c64-8930-683b11bd8531
# ╠═104b926e-5a36-41d7-a587-aeafa7bcd9c5
# ╟─eb4a78c4-0522-44ae-b3ab-fe489c135026
# ╠═1ae7dd74-40c5-439c-a95c-3d31eecc8d89
# ╠═a2939ec8-3ac1-4a7b-8e0b-68b15f0d98ce
# ╠═dd2ef429-7294-4602-895e-73031b3e8314
# ╠═aa891570-ad41-43d2-81c9-dd4303343f71
