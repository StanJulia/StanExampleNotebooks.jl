### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 1063debc-9e55-417d-9cc0-5b040a1f60af
using Pkg

# ╔═╡ 3dda1456-27a7-43b9-94df-8c1e66902ec1
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ aa89f177-03d8-409e-916e-208b9b897ea4
begin
	# Causal inference support
	using CausalInference

	# DAG graphics support
	using GraphViz
	using CairoMakie

	# Stan specific
	using StanSample
	
	# Project support libraries
	using StatisticalRethinking: sr_datadir
	using RegressionAndOtherStories
end

# ╔═╡ 7d41553e-779f-4cb5-8563-08b8afc4b7ed
md" ## Example chapter 1."

# ╔═╡ 6a8fe376-cf9d-4b39-a601-8f4763f6737b
md"##### Set page layout for notebook."

# ╔═╡ d3ce5ff1-3429-4a3b-93fd-9f246db34fe2
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(5px, 3%);
    	padding-right: max(5px, 25%);
	}
</style>
"""

# ╔═╡ 2e06da41-ff8d-4d2a-a96d-5d3cc1f48f5c
	tab = DataFrame(ag=1:5, mean_exer=[20, 25, 30, 35, 40],
		mean_chol=[80, 100, 120, 140, 160], slope_chol=[-1, -1.2, -1.4, -1.6, -1.8])


# ╔═╡ b3df5ef5-f979-44b0-a562-314566c2eed2
begin
	p = 0.01 # p-value for DAG independence tests
	N_men = 257
	N_women = 643
	df = DataFrame()
	age = Int.(round.(vcat(rand(Uniform(5, 54), N_men), rand(Uniform(5, 54), N_women))))
	age_groups = [10, 20, 30, 40, 50]
	age_group = repeat([0], N_men + N_women)
	age_group_ind = repeat([0], N_men + N_women)
	for (i, a) in enumerate(age)
		for (j, ag) in enumerate(age_groups)
			if (ag - 6) < a < (ag + 5)
				age_group[i] = ag
				age_group_ind[i] = j
			end
			continue
		end
	end
	df.age_group = age_group
	df.G = age_group_ind
	df.S = [i <= N_men ? 1 : 0 for i = 1:(N_men + N_women)]
	df.A = age
	exercise = Vector{Int}()
	cholesterol = Vector{Float64}()
	for r in eachrow(df)
		append!(exercise, round.(rand(Normal(tab.mean_exer[r.G], 4), 1)))
	end
	df.E = exercise
	for r in eachrow(df)
		append!(cholesterol, rand(Normal(tab.mean_chol[r.G], 6) + tab.slope_chol[r.G] * r.E, 1))
	end
	df.C = cholesterol
	df
end

# ╔═╡ 54efcd75-b375-4c05-ad44-679c48d40053
stan1_0 = "
data {
	int N;
	vector[N] C;
	vector[N] E;
}
parameters {
	real a;
	real b;
	real<lower=0> sigma;
}
model {
	vector[N] mu = a + b * E;
	a ~ normal(50, 10);
	b ~ normal(0, 10);
	C ~ normal(mu, sigma);
}
";

# ╔═╡ 2a04fe7f-4f42-426b-90ef-56cfc912bfbe
begin
	data = (N = size(df, 1), E = df.E, C = df.C, A = df.A, G = df.G, K = length(unique(df.G)))
	m1_0s = SampleModel("m1_0s", stan1_0)
	rc1_0s = stan_sample(m1_0s; data)
	success(rc1_0s) && describe(m1_0s, [:a, :b, :sigma])
end

# ╔═╡ 2a259fad-07f5-4698-9e91-e4c4c8ae601a
if success(rc1_0s)
	post1_0s_df = read_samples(m1_0s, :dataframe)
	ms1_0s = model_summary(post1_0s_df, [:a, :b, :sigma])
end

# ╔═╡ 7d8697e2-982b-4f43-9c5f-86cb937e91d7
let
	x = 10:50
	f = Figure(resolution=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="Exercise", ylabel="Cholesterol")
	scatter!(df.E, df.C)
	lines!(x, ms1_0s[:a, :mean] .+ ms1_0s[:b, :mean] .* x; color=:black)
	f
end

# ╔═╡ bfbd530f-e82b-42ad-9020-23f9f83fc91a
stan2_0 = "
data{
    int N;
	int K;
    vector[N] C;
    vector[N] E;
    array[N] int G;
}
parameters{
	vector[K] a;
    vector[K] b;
    real<lower=0> sigma;
}
model{
    vector[N] mu;
    sigma ~ exponential(1);
    a ~ normal( 100 , 10 );
	b ~ normal( -1 , 1 );
    for ( i in 1:N ) {
        mu[i] = a[G[i]] + b[G[i]] * E[i];
    }
    C ~ normal( mu , sigma );
}
";

# ╔═╡ f8b3e17e-479c-448a-bbf1-46530fb1c921
begin
	m2_0s = SampleModel("m2_0s", stan2_0)
	rc2_0s = stan_sample(m2_0s; data)
	success(rc2_0s) && describe(m2_0s, [Symbol("a[1]"), Symbol("a[2]"), Symbol("a[3]"), 
		Symbol("a[4]"), Symbol("a[5]"), Symbol("b[1]"), Symbol("b[2]"), Symbol("b[3]"), 
		Symbol("b[4]"), Symbol("b[5]"), :sigma])
end

# ╔═╡ ad19b534-6767-4304-b8ff-0d80c74a1c4c
begin
	post2_0s_df = read_samples(m2_0s, :dataframe)
	ms2_0s = model_summary(post2_0s_df, [Symbol("a.1"), Symbol("a.2"), Symbol("a.3"), 
		Symbol("a.4"), Symbol("a.5"), Symbol("b.1"), Symbol("b.2"), Symbol("b.3"), 
		Symbol("b.4"), Symbol("b.5"), :sigma])
end

# ╔═╡ 625fd2dd-b900-4926-9223-e5af60d7a6ce
let
	x = 10:50
	dft = df[df.G .== 3, :]
	x3 = minimum(dft.E):maximum(dft.E)
	m3 = mean(dft.E)
	s3 = mean
	
	f = Figure(resolution=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="Exercise", ylabel="Cholesterol",
		title="Overall regression line and age group 3 regression line")
	scatter!(df.E, df.C; color=df.G)
	lines!(x, ms1_0s[:a, :mean] .+ ms1_0s[:b, :mean] .* x; color=:black)
	lines!(x3, ms2_0s[Symbol("a.3"), :mean] .+ ms2_0s[Symbol("b.3"), :mean] .* x3; color=:grey)
	
	f

end

# ╔═╡ Cell order:
# ╟─7d41553e-779f-4cb5-8563-08b8afc4b7ed
# ╟─6a8fe376-cf9d-4b39-a601-8f4763f6737b
# ╠═d3ce5ff1-3429-4a3b-93fd-9f246db34fe2
# ╠═1063debc-9e55-417d-9cc0-5b040a1f60af
# ╠═3dda1456-27a7-43b9-94df-8c1e66902ec1
# ╠═aa89f177-03d8-409e-916e-208b9b897ea4
# ╠═2e06da41-ff8d-4d2a-a96d-5d3cc1f48f5c
# ╠═b3df5ef5-f979-44b0-a562-314566c2eed2
# ╠═54efcd75-b375-4c05-ad44-679c48d40053
# ╠═2a04fe7f-4f42-426b-90ef-56cfc912bfbe
# ╠═2a259fad-07f5-4698-9e91-e4c4c8ae601a
# ╠═7d8697e2-982b-4f43-9c5f-86cb937e91d7
# ╠═bfbd530f-e82b-42ad-9020-23f9f83fc91a
# ╠═f8b3e17e-479c-448a-bbf1-46530fb1c921
# ╠═ad19b534-6767-4304-b8ff-0d80c74a1c4c
# ╠═625fd2dd-b900-4926-9223-e5af60d7a6ce
