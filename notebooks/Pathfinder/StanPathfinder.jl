### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ 5084b8f0-65ac-4704-b1fc-2a9008132bd7
using Pkg

# ╔═╡ d7753cf6-7452-421a-a3ec-76e07646f808
Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 550371ad-d411-4e66-9d63-7329322c6ea1
begin
    # Specific to this notebook
	using StanSample
	using StanOptimize
    using StanPathfinder
	using Distributions
    using DataFrames
	using CairoMakie
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
tmpdir = joinpath(pwd(), "tmp")

# ╔═╡ edfb88bc-a66d-4035-af24-2b06e3253171
begin
	sm = SampleModel("bernoulli_sm", bernoulli_model)
	rc_sm = stan_sample(sm; data)
	if success(rc_sm)
		df_sm = read_samples(sm, :dataframe)
	end
	describe(df_sm)
end

# ╔═╡ c67f204e-1cbc-4fa9-924f-2d8e4da0cb87
d_normal = fit(Normal{Float64}, df_sm.theta)

# ╔═╡ 06fe53e6-bb72-40dc-ac55-4e5cb363f2a3
d_mle = fit_mle(Normal{Float64}, df_sm.theta)

# ╔═╡ 9d9e6493-07e4-44aa-8552-f9edc72a4142
begin
	om = OptimizeModel("bernoulli_om", bernoulli_model)
	rc_om = stan_optimize(om; data)
	if success(rc_om)
  		map, cnames = read_optimize(om)
	end
	map
end

# ╔═╡ cc7174d3-fa01-4ed7-a49e-7cd5ecd6d450
begin
	θ̂ = mean(map["theta"])
	d_map = Normal(θ̂, θ̂ * (1-θ̂))
end

# ╔═╡ a46d3d38-3c56-4b43-a7ee-57323dbb1e3b
let
	global pm2 = PathfinderModel("bernoulli_pf", bernoulli_model)
	rc = stan_pathfinder(pm2; data, num_chains=4, save_cmdstan_config=true)
	if all(success.(rc))
		a3d, colnames = read_pathfinder(pm2)
		global dfa = StanSample.convert_a3d(a3d, colnames, Val(:dataframes))
	end
end

# ╔═╡ 8b95d355-691a-47a5-8a10-ddac5fb3655a
println(StanPathfinder.cmdline(pm2, 1))

# ╔═╡ 886f5fee-07b2-4d7d-a7fd-3805570ce1b1
log_file=StanBase.log_file_path(pm2.output_base, 1)

# ╔═╡ c200e687-9967-4ebc-8b7b-6f1a14aadaa2
sample_file=StanBase.sample_file_path(pm2.output_base, 1)

# ╔═╡ 98a87e92-be9d-4735-9dba-b3b1d8f313d9
pm2.num_chains=4

# ╔═╡ 60490358-75a0-464e-b1e9-869219ceb2c9
pm2.num_threads

# ╔═╡ 037133fa-7c32-4bb7-993a-7251c63885dd
let
	str = read(joinpath(pm2.tmpdir, "$(pm2.name)_log_1.log"), String)
    findfirst("Path [", str)
    str = split(str[findfirst("Path [", str)[1]:end], "\n")[1:end-1]
end

# ╔═╡ d57e4518-c4ea-41c8-93bf-f34ecf35987c
let
	str = read(joinpath(pm2.tmpdir, "$(pm2.name)_log_2.log"), String)
    findfirst("Path [", str)
    str = split(str[findfirst("Path [", str)[1]:end], "\n")[1:end-1]
end

# ╔═╡ aa891570-ad41-43d2-81c9-dd4303343f71
res2 = create_pathfinder_profile_df(pm2)

# ╔═╡ fcc59e30-c156-42cb-8351-7d54b5c4bee9
let
	f = Figure(; size=(1000, 400))
	pm = PathfinderModel("bernoulli_plot", bernoulli_model)
	for i in 1:4
		ax = Axis(f[1, i]; title="Density theta (pathfinder run $i)\n 1000 psis samples")
		density!(dfa[i].theta)
	end
	for i in 1:4
		rc = stan_sample(sm; data)
		df_sm2 = read_samples(sm, :dataframe)
		ax = Axis(f[2, i]; title="Density theta (mcmc run $i)\n 4000 draws")
		density!(df_sm2.theta)
	end
	f
end

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
# ╠═edfb88bc-a66d-4035-af24-2b06e3253171
# ╠═c67f204e-1cbc-4fa9-924f-2d8e4da0cb87
# ╠═06fe53e6-bb72-40dc-ac55-4e5cb363f2a3
# ╠═9d9e6493-07e4-44aa-8552-f9edc72a4142
# ╠═cc7174d3-fa01-4ed7-a49e-7cd5ecd6d450
# ╠═a46d3d38-3c56-4b43-a7ee-57323dbb1e3b
# ╠═8b95d355-691a-47a5-8a10-ddac5fb3655a
# ╠═886f5fee-07b2-4d7d-a7fd-3805570ce1b1
# ╠═c200e687-9967-4ebc-8b7b-6f1a14aadaa2
# ╠═98a87e92-be9d-4735-9dba-b3b1d8f313d9
# ╠═60490358-75a0-464e-b1e9-869219ceb2c9
# ╠═037133fa-7c32-4bb7-993a-7251c63885dd
# ╠═d57e4518-c4ea-41c8-93bf-f34ecf35987c
# ╠═aa891570-ad41-43d2-81c9-dd4303343f71
# ╠═fcc59e30-c156-42cb-8351-7d54b5c4bee9
