### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f5ed29c5-7f99-4dd1-bfdc-34f4001b9c34
using Pkg

# ╔═╡ 283d88d1-0c3e-443e-bf44-63f03e869c12
begin
	# Causal inference support
	using CausalInference
	using KernelDensity

	# DAG graphics support
	using GraphViz
	using CairoMakie

	# Stan specific
	using StanSample
	
	# Project support libraries
	using StatisticalRethinking: sr_datadir
	using RegressionAndOtherStories
end

# ╔═╡ 7e562c21-86e0-4b22-8a2c-d0188c7a1ab8
md" ## Illustrating interventions via a toy example."

# ╔═╡ e8f64c1a-94cf-4c0e-8fe6-b31b25cbc16a
md" ##### Based on the blogs of Ferenc Huszár (https://www.inference.vc) and `The Book of Why`."

# ╔═╡ 18b36ec3-4a56-49d9-b0f2-c1d266adffb4
md"##### Set page layout for notebook."

# ╔═╡ 41619ae3-be1c-453f-9635-46be85e1e728
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(5px, 3%);
    	padding-right: max(5px, 35%);
	}
</style>
"""

# ╔═╡ fed755ad-e816-4659-9389-7ef4008c6456
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 6da32ca7-554b-463d-ba91-3ccd28d2dbbd
md" ### Three scripts."

# ╔═╡ 98d7c715-1fa0-434a-9498-12c54224647d
begin
	p = 0.01 # p-value for DAG independence tests
	N = 3000
	global df1 = DataFrame()
	x1 = randn(N)
	y1 = x1 .+ 1 .+ sqrt(3) .* randn(N)
	df1.x1 = x1
	df1.y1 .= y1
	y2 = 1 .+ 2 * randn(N)
	x2 = (y2 .- 1) ./ 4 .+ sqrt(3) .* randn(N) ./ 2
	df1.x2 = x2
	df1.y2 = y2
	x3 = randn(N)
	y3 = x3 .+ 1 .+ sqrt(3) .* randn(N)
	df1.x3 = x3
	df1.y3 = y3
	global X1 = [x1, x2, x3]
	global Y1 = [y1, y2, y3]
	df1
end

# ╔═╡ 06c69b9e-4f17-4331-85e0-48830ce8fe48
let
	fig = Figure(resolution=(900, 300))
	colors = [:darkblue, :darkred, :darkgreen]
	ranges = [1:5, 7:11, 13:17]
	for i in 1:3
		x = X1[i]
		y = Y1[i]
		ax = Axis(fig[1, ranges[i]])
		hist!(x, scale_to=0.6, bins=30, offset=1, direction=:y, color=colors[i])
		hidedecorations!(ax)
	    hidespines!(ax)
		ax = Axis(fig[2:6, ranges[i]]; xlabel="x", ylabel="y")
		xlims!(ax, -3, 3)
		ylims!(ax, -5, 7)
		scatter!(x, y, markersize=3, color=colors[i])
		ax = Axis(fig[2:6, i * 6])
		hist!(y, scale_to=0.6, bins=30, offset=1, direction=:x, color=colors[i])
		hidedecorations!(ax)
	    hidespines!(ax)
	end
	fig
end

# ╔═╡ b41a9dca-679b-4f94-a8d1-695ffe18489e
let
	N = 3000
	X = 3
	global df2 = DataFrame()
	x1 = repeat([X], N)
	y1 = x1 .+ 1 .+ sqrt(3) .* randn(N)
	x1 = repeat([3], N)
	x1 = jitter.(x1, 0.01)
	df2.x1 = x1
	df2.y1 .= y1
	
	y2 = 1 .+ 2 * randn(N)
	x2 = repeat([X], N)
	x2 = jitter.(x2, 0.01)
	df2.x2 = x2
	df2.y2 = y2
	
	z3 = randn(N)
	y3 = z3 .+ 1 .+ sqrt(3) .* randn(N)
	x3 = repeat([X], length(z3))
	x3 = jitter.(x3, 0.01)
	df2.x3 = x3
	df2.y3 = y3
	
	global X2 = [x1, x2, x3]
	global Y2 = [y1, y2, y3]
	df1
end

# ╔═╡ 3174b3ce-2482-4945-a1d4-bdb393656a8a
let
	fig = Figure(resolution=(900, 300))
	colors = [:blue, :green, :red]
	ranges = [1:5, 7:11, 13:17]
	for i in 1:3
		x = X2[i]
		y = Y2[i]
		ax = Axis(fig[1, ranges[i]])
		xlims!(ax, 2.5, 3.5)
		hist!(x, scale_to=0.6, bins=30, offset=1, direction=:y, color=colors[i])
		hidedecorations!(ax)
	    hidespines!(ax)
		ax = Axis(fig[2:6, ranges[i]]; xlabel="x", ylabel="y")
		xlims!(ax, 2.5, 3.5)
		ylims!(ax, -8, 10)
		scatter!(x, y, markersize=3, color=colors[i])
		ax = Axis(fig[2:6, i * 6])
		ylims!(ax, -8, 10)
		hist!(y, scale_to=0.6, bins=30, offset=1, direction=:x, color=colors[i])
		hidedecorations!(ax)
	    hidespines!(ax)
	end
	fig
end

# ╔═╡ 35a39a49-6708-4522-8209-958d2e034744
let
	f = Figure(resolution=default_figure_resolution)
	ax = Axis(f[1, 1]; title="p(y|do(X=3))")
	dens_y1 = density!(df2.y1; color = (:blue, 0.1), strokecolor = :blue, strokewidth = 3)
	dens_y2 = density!(df2.y2; color=(:green, 0.1), strokecolor = :green, strokewidth = 3)
	dens_y3 = density!(df2.y3; color=(:red, 0.1), strokecolor = :red, strokewidth = 3)

	Legend(f[1, 2],
    [dens_y1, dens_y2, dens_y3],
    ["p(y1|do(X=3))", "p(y2|do(X=3))", "p(y3|do(X=3))"])

	f
end

# ╔═╡ 1a6e2363-dedb-464d-b772-b34e49bd649d
let
	f = Figure(resolution=default_figure_resolution)
	ax = Axis(f[1, 1]; title="p(y|X=3)")
	dens_y1 = density!(df1.y1 .+ 3; color = (:blue, 0.1), strokecolor = :blue, strokewidth = 3)
	dens_y2 = density!(df1.y2 .+ 3; color=(:green, 0.1), strokecolor = :green, strokewidth = 3)
	dens_y3 = density!(df1.y3 .+ 3; color=(:red, 0.1), strokecolor = :red, strokewidth = 3)

	Legend(f[1, 2],
    [dens_y1, dens_y2, dens_y3],
    ["p(y1|X=3)", "p(y2|X=3)", "p(y3|X=3)"])

	f
end

# ╔═╡ c2b1ab39-48e7-44b9-85df-d22c22c4b391
md" ##### A quick look with an extension based on `CausalInference.jl`."

# ╔═╡ 9bf77cd3-31bc-49a1-8ab9-3e03071dadf4
dag_1 = create_fci_dag("dag_1", df1, "Digraph AMD {x1->y1;}");

# ╔═╡ 9fbe5dbb-793d-4ac7-8227-e86b5fc557b5
gvplot(dag_1)

# ╔═╡ 8a208443-094c-4aa2-8d13-ee752beb2b15
md" ##### From the data alone the FCI estimated causal graph is not conclusive."

# ╔═╡ 517cff73-1c92-45a2-a7cc-4e4f2bdae50b
md" ## Causal inference."

# ╔═╡ 10b10cf8-62f5-4e90-a291-964527394095
md" ### Use the example from `The book of why`. See figure 7.1."

# ╔═╡ d518ffac-dc0c-4d84-96f9-194d23486ee8
let
	N = 2000
	G = 0.4 .* rand(0:1, N)
	S = 0.4 .* rand(0:1, N) .+ 0.4 .* G
	#S = [i > 0.5 ? 1.0 : 0.0 for i in s]
	T = 0.2 .* rand(Uniform(0, 1), N) .+ 0.4 .* S
	c = 0.2 .* rand(Uniform(0, 1), N) .+ 0.4 .* T .+ 0.4 .* G
	C =  [i > 0.3 ? 1.0 : 0.0 for i in c]
	global df = DataFrame(S=S, T=T, C=C)
	global df_full = DataFrame(G=G, S=S, T=T, C=C)
end

# ╔═╡ b5174b8a-2035-4a58-84aa-604620809500
p_s = sum(df.S) / nrow(df)

# ╔═╡ 444e15fe-1424-4b4a-bc03-dbb719af2bb7
p_c = sum(df.C) / nrow(df)

# ╔═╡ 790b3c1f-c3ea-47bf-933f-8240fda1a19b
md" ###### What are `p_smoking_when_cancer` and `p_cancer_when_smoking`?"

# ╔═╡ 0650018d-021b-4063-9ea3-9ae082b738b2
findall(x -> x ==1, df.C)

# ╔═╡ 6e271a8a-ee66-4b3d-9f27-a1b1ecaca547
findall(x -> x > 0.1, df.S)

# ╔═╡ c047c97c-956f-4f8a-b3f2-1bd1dc449d37
p_s_c = sum(df.C[intersect(findall(x -> x == 1, df.C), findall(x -> x > 0.4, df.S))]) / nrow(df)

# ╔═╡ 767bd60e-f7c3-4970-91b7-049622514249
p_s_g_c = p_s_c / p_c

# ╔═╡ 348a800d-df39-4e9d-9abd-b9dfbcf7ab6d
p_c_g_s = p_s_c / p_s

# ╔═╡ 9aad2b06-29eb-44ed-bfbf-6636630a112f
likelihood_ratio_c = p_c_g_s / p_c

# ╔═╡ 6b2ed183-0894-4212-9d62-67f0b6f41e0e
likelihood_ratio_s = p_s_g_c / p_s

# ╔═╡ 7e559bec-63f1-45bf-a9c5-27e788d2a239
describe(df_full)

# ╔═╡ ff23ec16-7082-4d47-acf1-0bdf5093a855
dag_2 = create_fci_dag("dag_2", df_full, "Digraph AMD {G->S; S->T; T->C; G->C;}");

# ╔═╡ d4ab128e-ffa2-4691-8c41-088fb678634e
gvplot(dag_2)

# ╔═╡ f57ecc39-c721-4bda-ad9f-d9b09617dd48
backdoor_criterion(dag_2, :S, :C)

# ╔═╡ cadcb7b4-77fe-406b-84dd-8e0695dd9f27
md" ##### There is a backdoor path."

# ╔═╡ 3703ee5f-f0d4-4986-b791-51eaa072cce0
all_paths(dag_2, :S, :C)

# ╔═╡ 97381b10-ca9c-42d1-89b9-59909a4db3ec
backdoor_criterion(dag_2, :S, :C, [:G])

# ╔═╡ 1ef91f8b-a2ed-404c-8487-f752971c800c
dsep(dag_2, :S, :C, [:T, :G])

# ╔═╡ debc67b5-c816-4c66-a15e-7511c2363b82
md" ##### G is not observed, so we can't condition on G."

# ╔═╡ ebe611c1-31e3-4e37-b165-acfef299bf0a
backdoor_criterion(dag_2, :T, :C, [:S])

# ╔═╡ 652ad04a-b38c-4254-9d86-55079403d848
dag_3 = create_fci_dag("dag_3", df, "Digraph AMD {S->T; T->C;}",);

# ╔═╡ c75fdeab-eccd-4731-a3eb-55e20053613b
gvplot(dag_3; title_g="Observed part of generational model")

# ╔═╡ c8b29c82-0820-4669-9926-3f3778c57b36
stan2_1 = "
data {
	int N;
	vector[N] S;
	vector[N] T;
	vector[N] C;
}
parameters {
	real a;
	real bS;
	real<lower=0> sigma;
}
model {
	vector[N] mu = a + bS * S;
	a ~ normal(0, 3);
	bS ~ normal(0, 3);
	sigma ~ exponential(1);
	C ~ normal(mu, sigma);
}";

# ╔═╡ 5c51c997-d906-4a40-b9a7-5e713945c865
stan2_2 = "
data {
	int N;
	vector[N] S;
	vector[N] T;
	vector[N] C;
}
parameters {
	real a;
	real bT;
	real<lower=0> sigma;
}
model {
	vector[N] mu = a + bT * T;
	a ~ normal(0, 3);
	bT ~ normal(0, 3);
	sigma ~ exponential(1);
	C ~ normal(mu, sigma);
}";

# ╔═╡ a3c910a2-9204-4d23-a553-1daf48c9857b
stan2_3 = "
data {
	int N;
	vector[N] S;
	vector[N] T;
	vector[N] C;
}
parameters {
	real a;
	real bS;
	real bT;
	real<lower=0> sigma;
}
model {
	vector[N] mu = a + bT * T + bS * S;
	a ~ normal(0, 3);
	bS ~ normal(0, 3);
	bT ~ normal(0, 3);
	sigma ~ exponential(1);
	C ~ normal(mu, sigma);
}";

# ╔═╡ 7c333eb8-1ce7-4d87-b9c9-9d2c93dba78c
stan2_4 = "
data {
	int N;
	vector[N] S;
	vector[N] C;
	vector[N] G;
}
parameters {
	real a;
	real bS;
	real bG;
	real<lower=0> sigma;
}
model {
	vector[N] mu = a + bS * S + bG * G;
	a ~ normal(0, 3);
	bS ~ normal(0, 3);
	bG ~ normal(0, 3);
	sigma ~ exponential(1);
	C ~ normal(mu, sigma);
}";

# ╔═╡ a4d4c1ec-9c99-4d8d-970d-0a275570eba9
let
	data = (N = nrow(df), S = df.S, T = df.T, C=df.C)
	global m2_1s = SampleModel("m2_1s", stan2_1)
	global rc2_1s = stan_sample(m2_1s; data)
	success(rc2_1s) && describe(m2_1s, [:a, :bS, :sigma])
end

# ╔═╡ 147a2c90-14d4-40b7-8620-cbf2a0e9307c
let
	data = (N = nrow(df), S = df.S, T = df.T, C=df.C)
	global m2_2s = SampleModel("m2_2s", stan2_2)
	global rc2_2s = stan_sample(m2_2s; data)
	success(rc2_2s) && describe(m2_2s, [:a, :bT, :sigma])
end

# ╔═╡ a298db41-329b-491b-a9e2-a02bfd3a2a5d
let
	data = (N = nrow(df), S = df.S, T = df.T, C=df.C)
	global m2_3s = SampleModel("m2_3s", stan2_3)
	global rc2_3s = stan_sample(m2_3s; data)
	success(rc2_3s) && describe(m2_3s, [:a, :bS, :bT, :sigma])
end

# ╔═╡ bfac56eb-07b9-4c4b-a472-88f28003424e
let
	data = (N = nrow(df_full), S = df_full.S, G = df_full.G, C=df_full.C)
	global m2_4s = SampleModel("m2_4s", stan2_4)
	global rc2_4s = stan_sample(m2_4s; data)
	success(rc2_4s) && describe(m2_4s, [:a, :bS, :bG, :sigma])
end

# ╔═╡ 39a8e527-41cb-446a-86ec-9006791e3129
if success(rc2_1s)
	post2_1s_df = read_samples(m2_1s, :dataframe)
	ms2_1s_df = model_summary(post2_1s_df, 
		[:a, :bS, :sigma])
end

# ╔═╡ ddfb1365-26d3-4bf4-a161-c79a1b4f19f4
let
	if success(rc2_1s) && success(rc2_2s) && success(rc2_3s)
		(s1, f1) = plot_model_coef([m2_1s, m2_2s, m2_3s], 
			[:a, :bS, :bT, :sigma]; 
			title="Summary of coefficients in models m2_1s, m2_2s, m2_3s.")
		f1
	end
end

# ╔═╡ 27465511-9fc1-4871-9e08-f9d21f04f560
let
	if success(rc2_1s) && success(rc2_4s)
		(s1, f1) = plot_model_coef([m2_1s, m2_4s], 
			[:a, :bG, :bS, :sigma]; 
			title="Summary of coefficients in models m2_1s and m2_4s.")
		f1
	end
end

# ╔═╡ Cell order:
# ╟─7e562c21-86e0-4b22-8a2c-d0188c7a1ab8
# ╟─e8f64c1a-94cf-4c0e-8fe6-b31b25cbc16a
# ╟─18b36ec3-4a56-49d9-b0f2-c1d266adffb4
# ╠═41619ae3-be1c-453f-9635-46be85e1e728
# ╠═f5ed29c5-7f99-4dd1-bfdc-34f4001b9c34
# ╠═fed755ad-e816-4659-9389-7ef4008c6456
# ╠═283d88d1-0c3e-443e-bf44-63f03e869c12
# ╟─6da32ca7-554b-463d-ba91-3ccd28d2dbbd
# ╠═98d7c715-1fa0-434a-9498-12c54224647d
# ╠═06c69b9e-4f17-4331-85e0-48830ce8fe48
# ╠═b41a9dca-679b-4f94-a8d1-695ffe18489e
# ╠═3174b3ce-2482-4945-a1d4-bdb393656a8a
# ╠═35a39a49-6708-4522-8209-958d2e034744
# ╠═1a6e2363-dedb-464d-b772-b34e49bd649d
# ╟─c2b1ab39-48e7-44b9-85df-d22c22c4b391
# ╠═9bf77cd3-31bc-49a1-8ab9-3e03071dadf4
# ╠═9fbe5dbb-793d-4ac7-8227-e86b5fc557b5
# ╟─8a208443-094c-4aa2-8d13-ee752beb2b15
# ╟─517cff73-1c92-45a2-a7cc-4e4f2bdae50b
# ╠═10b10cf8-62f5-4e90-a291-964527394095
# ╠═d518ffac-dc0c-4d84-96f9-194d23486ee8
# ╠═b5174b8a-2035-4a58-84aa-604620809500
# ╠═444e15fe-1424-4b4a-bc03-dbb719af2bb7
# ╠═790b3c1f-c3ea-47bf-933f-8240fda1a19b
# ╠═0650018d-021b-4063-9ea3-9ae082b738b2
# ╠═6e271a8a-ee66-4b3d-9f27-a1b1ecaca547
# ╠═c047c97c-956f-4f8a-b3f2-1bd1dc449d37
# ╠═767bd60e-f7c3-4970-91b7-049622514249
# ╠═348a800d-df39-4e9d-9abd-b9dfbcf7ab6d
# ╠═9aad2b06-29eb-44ed-bfbf-6636630a112f
# ╠═6b2ed183-0894-4212-9d62-67f0b6f41e0e
# ╠═7e559bec-63f1-45bf-a9c5-27e788d2a239
# ╠═ff23ec16-7082-4d47-acf1-0bdf5093a855
# ╠═d4ab128e-ffa2-4691-8c41-088fb678634e
# ╠═f57ecc39-c721-4bda-ad9f-d9b09617dd48
# ╟─cadcb7b4-77fe-406b-84dd-8e0695dd9f27
# ╠═3703ee5f-f0d4-4986-b791-51eaa072cce0
# ╠═97381b10-ca9c-42d1-89b9-59909a4db3ec
# ╠═1ef91f8b-a2ed-404c-8487-f752971c800c
# ╟─debc67b5-c816-4c66-a15e-7511c2363b82
# ╠═ebe611c1-31e3-4e37-b165-acfef299bf0a
# ╠═652ad04a-b38c-4254-9d86-55079403d848
# ╠═c75fdeab-eccd-4731-a3eb-55e20053613b
# ╠═c8b29c82-0820-4669-9926-3f3778c57b36
# ╠═5c51c997-d906-4a40-b9a7-5e713945c865
# ╠═a3c910a2-9204-4d23-a553-1daf48c9857b
# ╠═7c333eb8-1ce7-4d87-b9c9-9d2c93dba78c
# ╠═a4d4c1ec-9c99-4d8d-970d-0a275570eba9
# ╠═147a2c90-14d4-40b7-8620-cbf2a0e9307c
# ╠═a298db41-329b-491b-a9e2-a02bfd3a2a5d
# ╠═bfac56eb-07b9-4c4b-a472-88f28003424e
# ╠═39a8e527-41cb-446a-86ec-9006791e3129
# ╠═ddfb1365-26d3-4bf4-a161-c79a1b4f19f4
# ╠═27465511-9fc1-4871-9e08-f9d21f04f560
