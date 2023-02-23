### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 62c80a26-975a-11ed-2e09-2dce0e33bb70
using Pkg

# ╔═╡ aaea31c8-37ed-4f0f-8e3e-8e89d30ed918
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 58ece6dd-a20f-4624-898a-40cae4b471e4
begin
	# General packages for this script
	using PlutoUI
	using Test
	
	# Graphics related packages
	using GLMakie
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
dag_1 = dag("Dag_1", [(:X, :V), (:X, :W), (:W, :Z), (:V, :Z), (:Z, :S)]; df=df1, covm=covm1);

# ╔═╡ 728aa5ff-d581-43f7-bacf-c04787480bb7
gvplot(dag_1)

# ╔═╡ 30c7782a-4a56-4971-a14d-bd96104140cf
dag_1.v

# ╔═╡ a0cc8175-4f83-45f8-8bba-3e0679ff4ccb
dseparation(dag_1, :X, :V)

# ╔═╡ 00ad74d9-62d8-4ced-8bf1-eace47470272
dseparation(dag_1, :X, :Z, [:W], verbose=true)

# ╔═╡ 5533711c-6cbb-4407-8081-1ab44a09a8b9
dseparation(dag_1, :X, :Z, [:V], verbose=true)

# ╔═╡ 6d999053-3612-4e8d-b2f2-2ddf3eae5630
dseparation(dag_1, :X, :Z, [:V, :W], verbose=true)

# ╔═╡ d94d4717-7ca8-4db9-ae54-fc481aa63c3c
@time est_dag_1_g = pcalg(df1, p, gausscitest)

# ╔═╡ 603619ac-3714-46ab-8106-cceacb2dc11c
est_dag_1 = dag("est_dag_1", est_dag_1_g, dag_1);

# ╔═╡ e955f2f6-91c5-476e-9aae-96163de036e4
gvplot(est_dag_1)

# ╔═╡ 240e4e7b-b90f-4dd8-9862-e3cf4876ab98
est_dag_1.covm

# ╔═╡ daa82ef0-fd80-4ece-ba12-93a47a30a176
@time est_g_1_2 = pcalg(df1, p, cmitest)

# ╔═╡ f3007639-619a-4153-bbb2-7699e08a8c1e
est_dag_1_2 = dag("est_dag_1", est_g_1_2, dag_1);

# ╔═╡ 0d49b778-1df2-47cf-add2-4bebd2ef6bc8
gvplot((est_dag_1_2))

# ╔═╡ 655261b4-5b53-4b3f-952e-6c2b09dea2be
md" ##### Play around with the estimated graph when varying X -> W."

# ╔═╡ ad72f618-c6b4-403c-926e-31dbab1c2c1c
@bind bXW PlutoUI.Slider(0:100; default=25)

# ╔═╡ 417e1430-7593-4755-abdc-69ba7059a3db
bXW

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

# ╔═╡ d0da3670-1e10-4cd7-bef7-cfa185572359
let
	Random.seed!(123)
	N = 1000
	x = rand(N)
	v = x + rand(N) * 0.25
	w = x * bXW * 0.01 + rand(N) * 0.25
	z = v + w + rand(N) * 0.25
	s = z + rand(N) * 0.25
	global X4 = [x v w z s]
	global df4 = DataFrame(x=x, v=v, w=w, z=z, s=s)
	global covm4 = NamedArray(cov(Array(df2)), (names(df2), names(df2)), ("Rows", "Cols"))
end;

# ╔═╡ 810fde77-991d-495a-9966-50c1a7abc685
dag_4 = dag("Dag_4", [(:X, :V), (:X, :W), (:W, :Z), (:V, :Z), (:Z, :S)]; df=df4, covm=covm4);

# ╔═╡ 124d0947-1a6f-4e01-aa6f-a96098720cb5
begin
	@time est_dag_4_g = pcalg(df4, p, gausscitest)
	est_dag_4 = dag("est_dag_2", est_dag_4_g, dag_1)
	gvplot(est_dag_4)
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

# ╔═╡ 9105868e-8a7c-457e-8760-ab5cfcb62955
levels = length([m5_5_1s, m5_5_2s, m5_5_3s]) * (length([:a, :bV, :bW]) + 1)

# ╔═╡ 307076f5-fd19-41af-84b3-054609060d9d
gvplot(est_dag_1)

# ╔═╡ ea512f88-d6dd-4e96-8716-f8b9ee017152
begin
	@time est_dag_2_g = pcalg(df2, p, gausscitest)
	est_dag_2 = dag("est_dag_2", est_dag_2_g, dag_1)
	gvplot(est_dag_2)
end

# ╔═╡ 7d74e01c-dcdc-4fc3-bc18-06563ffc7573
begin
	@time est_dag_3_g = pcalg(df3, p, gausscitest)
	est_dag_3 = dag("est_dag_3", est_dag_3_g, dag_1)
	gvplot(est_dag_3)
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
# ╠═728aa5ff-d581-43f7-bacf-c04787480bb7
# ╠═30c7782a-4a56-4971-a14d-bd96104140cf
# ╠═a0cc8175-4f83-45f8-8bba-3e0679ff4ccb
# ╠═00ad74d9-62d8-4ced-8bf1-eace47470272
# ╠═5533711c-6cbb-4407-8081-1ab44a09a8b9
# ╠═6d999053-3612-4e8d-b2f2-2ddf3eae5630
# ╠═d94d4717-7ca8-4db9-ae54-fc481aa63c3c
# ╠═603619ac-3714-46ab-8106-cceacb2dc11c
# ╠═e955f2f6-91c5-476e-9aae-96163de036e4
# ╠═240e4e7b-b90f-4dd8-9862-e3cf4876ab98
# ╠═daa82ef0-fd80-4ece-ba12-93a47a30a176
# ╠═f3007639-619a-4153-bbb2-7699e08a8c1e
# ╠═0d49b778-1df2-47cf-add2-4bebd2ef6bc8
# ╟─655261b4-5b53-4b3f-952e-6c2b09dea2be
# ╠═ad72f618-c6b4-403c-926e-31dbab1c2c1c
# ╠═417e1430-7593-4755-abdc-69ba7059a3db
# ╠═d0da3670-1e10-4cd7-bef7-cfa185572359
# ╠═810fde77-991d-495a-9966-50c1a7abc685
# ╠═124d0947-1a6f-4e01-aa6f-a96098720cb5
# ╟─fad832c7-ec30-4382-9555-cd1ddfd6a909
# ╠═8c7df65a-88fe-47ba-8ed5-39f469ade8aa
# ╠═75a21a67-dfb6-4fba-b356-f611c403adab
# ╠═46aebc7b-2b2e-4254-af16-73ee2512ad5f
# ╠═5bb2fec8-b915-4aff-855b-0a3d325de66b
# ╠═6de18959-babc-4f56-af02-595a7cc7fdc6
# ╠═9105868e-8a7c-457e-8760-ab5cfcb62955
# ╠═307076f5-fd19-41af-84b3-054609060d9d
# ╠═ea512f88-d6dd-4e96-8716-f8b9ee017152
# ╠═7d74e01c-dcdc-4fc3-bc18-06563ffc7573
