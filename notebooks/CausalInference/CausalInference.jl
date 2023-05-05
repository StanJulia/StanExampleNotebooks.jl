### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 62c80a26-975a-11ed-2e09-2dce0e33bb70
using Pkg

# ╔═╡ 58ece6dd-a20f-4624-898a-40cae4b471e4
begin
	# General packages for this script
	using Test
	
	# Graphics related packages
	using CairoMakie
	using GraphViz
	using Graphs

	# DAG support
	using CausalInference

	# Stan specific
	using StanSample

	# Project support functions
	using StatisticalRethinking: sr_datadir
	using RegressionAndOtherStories
end

# ╔═╡ e4552c81-d0db-4434-b81a-c86f1af515e5
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(10px, 5%);
    	padding-right: max(10px, 36%);
	}
</style>
"""

# ╔═╡ aaea31c8-37ed-4f0f-8e3e-8e89d30ed918
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 261cca70-a6dd-4bed-b2f2-8667534d0ceb
let
	Random.seed!(123)
	N = 1000
	global p = 0.01
	x = rand(N)
	v = x + rand(N) * 0.25
	w = x + rand(N) * 0.25
	z = v + w * 0.5 + rand(N) * 0.25
	s = z + rand(N) * 0.25
	global X1 = [x v w z s]
	global df1 = DataFrame(x=x, v=v, w=w, z=z, s=s)
	global covm1 = NamedArray(cov(Array(df1)), (names(df1), names(df1)), ("Rows", "Cols"))
	df1
end

# ╔═╡ 1608578d-32b0-4b6e-848e-dca8417e1099
stan5_5 = "
data {
  int N;
  vector[N] X;
  vector[N] W;
  vector[N] V;
  vector[N] Z;
}
parameters {
  real a;
  real bV;
  real bW;
  real aW;
  real bXW;
  real<lower=0> sigma;
  real<lower=0> sigma_W;
}
model {
  vector[N] mu = a + bV * V + bW * W;
  a ~ normal( 0 , 0.2 );
  bV ~ normal( 0 , 0.5 );
  bW ~ normal( 0 , 0.5 );
  sigma ~ exponential( 1 );
  Z ~ normal( mu , sigma );
  // X -> W
  vector[N] mu_W = aW + bXW * X;
  aW ~ normal( 0 , 0.5 );
  bXW ~ normal( 0 , 0.5 );
  sigma_W ~ exponential( 1 );
  W ~ normal( mu_W , sigma_W );
}
";

# ╔═╡ 1fa40de3-9423-43af-ae3c-78f89f9897bb
let
	# X -> V -> Z <- W <- X
	data = (N = size(df1, 1), X = df1.x, V = df1.v, W = df1.w, Z = df1.z)
	global m5_5_1s = SampleModel("m5.5.1s", stan5_5)
	global rc5_5_1s = stan_sample(m5_5_1s; data)
	success(rc5_5_1s) && describe(m5_5_1s, [:a, :bV, :bW, :sigma, :aW, :bXW, :sigma_W])
end

# ╔═╡ ad08dd09-222a-4071-92d4-38deebaf2e82
md" ### CausalInference.jl"

# ╔═╡ 6bbfe4cb-f7e1-4503-a386-092882a1a49c
begin
	dag_1_dot_str = "DiGraph Dag_1 {x -> v; x -> w; w -> z; v -> z; z -> s;}"
	dag_1 = create_dag("Dag_1", df1; g_dot_str=dag_1_dot_str)
	gvplot(dag_1)
end

# ╔═╡ 2e52ff5a-41de-4cca-918c-f1c9b2be9e2e
dag_1.vars

# ╔═╡ 49bb18f3-d914-4600-9750-6a0478712851
dag_1.g

# ╔═╡ f6edfa9e-6405-4fcf-a2bc-e05f88acff9d
dag_1.est_vars

# ╔═╡ b71c9a06-2ecd-40f3-95f7-948a73290d40
dag_1.est_g

# ╔═╡ d0360ab1-afea-41a6-95ca-a43468d632ce
dag_1.est_g_dot_str

# ╔═╡ a0cc8175-4f83-45f8-8bba-3e0679ff4ccb
dsep(dag_1, :x, :v)

# ╔═╡ 00ad74d9-62d8-4ced-8bf1-eace47470272
dsep(dag_1, :x, :z, [:w], verbose=true)

# ╔═╡ 5533711c-6cbb-4407-8081-1ab44a09a8b9
dsep(dag_1, :x, :z, [:v], verbose=true)

# ╔═╡ 6d999053-3612-4e8d-b2f2-2ddf3eae5630
dsep(dag_1, :x, :z, [:v, :w], verbose=true)

# ╔═╡ d94d4717-7ca8-4db9-ae54-fc481aa63c3c
@time est_dag_1_g = pcalg(df1, p, gausscitest)

# ╔═╡ e250a4cc-65a5-4c86-bffb-987f664f12c8
md" ##### Compare with FCI algorithm results."

# ╔═╡ c78f95b1-5015-4ae6-ba21-8bea7a7b8772
g_oracle = fcialg(5, dseporacle, dag_1.g)

# ╔═╡ c60bb1fb-3f0f-4ee6-884b-994f994788b0
g_gauss = fcialg(dag_1.df, 0.05, gausscitest)

# ╔═╡ 0db37602-6997-474a-8cc1-0b3bc8a6fb40
let
    fci_oracle_dot_str = to_gv(g_oracle, dag_1.vars)
    fci_gauss_dot_str = to_gv(g_gauss, dag_1.vars)
    g1 = GraphViz.Graph(dag_1.g_dot_str)
    g2 = GraphViz.Graph(dag_1.est_g_dot_str)
    g3 = GraphViz.Graph(fci_oracle_dot_str)
    g4 = GraphViz.Graph(fci_gauss_dot_str)
    f = Figure(resolution=default_figure_resolution)
    ax = Axis(f[1, 1]; aspect=DataAspect(), title="True (generational) DAG")
    CairoMakie.image!(rotr90(create_png_image(g1)))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(f[1, 2]; aspect=DataAspect(), title="PC estimated DAG")
    CairoMakie.image!(rotr90(create_png_image(g2)))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(f[2, 1]; aspect=DataAspect(), title="FCI oracle estimated DAG")
    CairoMakie.image!(rotr90(create_png_image(g3)))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(f[2, 2]; aspect=DataAspect(), title="FCI gauss estimated DAG")
    CairoMakie.image!(rotr90(create_png_image(g4)))
    hidedecorations!(ax)
    hidespines!(ax)
    f
end

# ╔═╡ fad832c7-ec30-4382-9555-cd1ddfd6a909
md" ##### Play around with the effect of varying the strength of Z -> W."

# ╔═╡ 8c7df65a-88fe-47ba-8ed5-39f469ade8aa
let
	Random.seed!(123)
	N = 1000
	p = 0.01
	x = rand(N)
	v = x + rand(N) * 0.25
	w = x + rand(N) * 0.25
	z = v + w * 0.1 + rand(N) * 0.25
	s = z + rand(N) * 0.25

	global X2 = [x v w z s]
	global df2 = DataFrame(x=x, v=v, w=w, z=z, s=s)
	global covm2 = NamedArray(cov(Array(df2)), (names(df2), names(df2)), ("Rows", "Cols"))
	df2
end

# ╔═╡ 75a21a67-dfb6-4fba-b356-f611c403adab
# ╠═╡ show_logs = false
let
	# X -> V -> Z <- W <- X
	data = (N = size(df2, 1), X = df2.x, V = df2.v, W = df2.w, Z = df2.z)
	global m5_5_2s = SampleModel("m5.5.2s", stan5_5)
	global rc5_5_2s = stan_sample(m5_5_2s; data)
	success(rc5_5_2s) && describe(m5_5_2s, [:a, :bV, :bW, :sigma, :aW, :bXW, :sigma_W])
end

# ╔═╡ 46aebc7b-2b2e-4254-af16-73ee2512ad5f
let
	Random.seed!(123)
	N = 1000
	p = 0.01
	x = rand(N)
	v = x + rand(N) * 0.25
	w = x + rand(N) * 0.25
	z = v + w * 0.01 + rand(N) * 0.25
	s = z + rand(N) * 0.25
	global X3 = [x v w z s]
	global df3 = DataFrame(x=x, v=v, w=w, z=z, s=s)
	global covm3 = NamedArray(cov(Array(df3)), (names(df3), names(df3)), ("Rows", "Cols"))
	df3
end

# ╔═╡ 5bb2fec8-b915-4aff-855b-0a3d325de66b
let
	# X -> V -> Z <- W <- X
	data = (N = size(df3, 1), X = df3.x, V = df3.v, W = df3.w, Z = df3.z)
	global m5_5_3s = SampleModel("m5.5.3s", stan5_5)
	global rc5_5_3s = stan_sample(m5_5_3s; data)
	success(rc5_5_3s) && describe(m5_5_3s, [:a, :bV, :bW, :sigma, :aW, :bXW, :sigma_W])
end

# ╔═╡ 6de18959-babc-4f56-af02-595a7cc7fdc6
if success(rc5_5_1s) && success(rc5_5_2s) && success(rc5_5_3s) 
	(s1, f1) = plot_model_coef([m5_5_1s, m5_5_2s, m5_5_3s], [:a, :bV, :bW, :sigma]; 
		title="Summary of coefficients in models m5_5_1s, m5_5_2s and m5_5_3s.")
	f1
end

# ╔═╡ Cell order:
# ╠═e4552c81-d0db-4434-b81a-c86f1af515e5
# ╠═62c80a26-975a-11ed-2e09-2dce0e33bb70
# ╠═aaea31c8-37ed-4f0f-8e3e-8e89d30ed918
# ╠═58ece6dd-a20f-4624-898a-40cae4b471e4
# ╠═261cca70-a6dd-4bed-b2f2-8667534d0ceb
# ╠═1608578d-32b0-4b6e-848e-dca8417e1099
# ╠═1fa40de3-9423-43af-ae3c-78f89f9897bb
# ╟─ad08dd09-222a-4071-92d4-38deebaf2e82
# ╠═6bbfe4cb-f7e1-4503-a386-092882a1a49c
# ╠═2e52ff5a-41de-4cca-918c-f1c9b2be9e2e
# ╠═49bb18f3-d914-4600-9750-6a0478712851
# ╠═f6edfa9e-6405-4fcf-a2bc-e05f88acff9d
# ╠═b71c9a06-2ecd-40f3-95f7-948a73290d40
# ╠═d0360ab1-afea-41a6-95ca-a43468d632ce
# ╠═a0cc8175-4f83-45f8-8bba-3e0679ff4ccb
# ╠═00ad74d9-62d8-4ced-8bf1-eace47470272
# ╠═5533711c-6cbb-4407-8081-1ab44a09a8b9
# ╠═6d999053-3612-4e8d-b2f2-2ddf3eae5630
# ╠═d94d4717-7ca8-4db9-ae54-fc481aa63c3c
# ╟─e250a4cc-65a5-4c86-bffb-987f664f12c8
# ╠═c78f95b1-5015-4ae6-ba21-8bea7a7b8772
# ╠═c60bb1fb-3f0f-4ee6-884b-994f994788b0
# ╠═0db37602-6997-474a-8cc1-0b3bc8a6fb40
# ╟─fad832c7-ec30-4382-9555-cd1ddfd6a909
# ╠═8c7df65a-88fe-47ba-8ed5-39f469ade8aa
# ╠═75a21a67-dfb6-4fba-b356-f611c403adab
# ╠═46aebc7b-2b2e-4254-af16-73ee2512ad5f
# ╠═5bb2fec8-b915-4aff-855b-0a3d325de66b
# ╠═6de18959-babc-4f56-af02-595a7cc7fdc6
