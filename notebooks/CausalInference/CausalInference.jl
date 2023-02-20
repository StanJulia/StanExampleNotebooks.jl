### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 62c80a26-975a-11ed-2e09-2dce0e33bb70
using Pkg

# ╔═╡ aaea31c8-37ed-4f0f-8e3e-8e89d30ed918
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 58ece6dd-a20f-4624-898a-40cae4b471e4
begin
	# General packages for this script
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
	z = v + w + rand(N) * 0.25
	s = z + rand(N) * 0.25

	global X = [x v w z s]
	global df = DataFrame(x=x, v=v, w=w, z=z, s=s)
	global covm = NamedArray(cov(Array(df)), (names(df), names(df)), ("Rows", "Cols"))
	df
end

# ╔═╡ ad08dd09-222a-4071-92d4-38deebaf2e82
md" ### CausalInference.jl"

# ╔═╡ 6bbfe4cb-f7e1-4503-a386-092882a1a49c
dag_1 = dag("Dag_1", [(:X, :V), (:X, :W), (:W, :Z), (:V, :Z), (:Z, :S)]; df, covm);

# ╔═╡ 728aa5ff-d581-43f7-bacf-c04787480bb7
gvplot(dag_1)

# ╔═╡ 30c7782a-4a56-4971-a14d-bd96104140cf
dag_1.v

# ╔═╡ a0cc8175-4f83-45f8-8bba-3e0679ff4ccb
dseparation(dag_1, :X, :V)

# ╔═╡ 00ad74d9-62d8-4ced-8bf1-eace47470272
dseparation(dag_1, :X, :S, [:W], verbose=true)

# ╔═╡ 5533711c-6cbb-4407-8081-1ab44a09a8b9
dseparation(dag_1, :X, :S, [:Z], verbose=true)

# ╔═╡ 6d999053-3612-4e8d-b2f2-2ddf3eae5630
dseparation(dag_1, :X, :Z, [:V, :W], verbose=true)

# ╔═╡ d94d4717-7ca8-4db9-ae54-fc481aa63c3c
@time est_g = pcalg(df, p, gausscitest)

# ╔═╡ 603619ac-3714-46ab-8106-cceacb2dc11c
dag_2 = dag("est_dag_1", est_g, dag_1);

# ╔═╡ 883a4e03-aa64-4b1c-b841-2a7eb73f4771
dag_1.d

# ╔═╡ 2fc0dae8-bd25-4323-bd8f-d0d27f9eb06d
dag_2.d

# ╔═╡ e955f2f6-91c5-476e-9aae-96163de036e4
gvplot(dag_2)

# ╔═╡ 240e4e7b-b90f-4dd8-9862-e3cf4876ab98
dag_2.covm

# ╔═╡ daa82ef0-fd80-4ece-ba12-93a47a30a176
@time est_g_2 = pcalg(df, p, cmitest)

# ╔═╡ f3007639-619a-4153-bbb2-7699e08a8c1e
est_dag_2 = dag("est_dag_2", est_g, dag_1);

# ╔═╡ 0d49b778-1df2-47cf-add2-4bebd2ef6bc8
gvplot((est_dag_2))

# ╔═╡ Cell order:
# ╠═e4552c81-d0db-4434-b81a-c86f1af515e5
# ╠═62c80a26-975a-11ed-2e09-2dce0e33bb70
# ╠═aaea31c8-37ed-4f0f-8e3e-8e89d30ed918
# ╠═58ece6dd-a20f-4624-898a-40cae4b471e4
# ╠═261cca70-a6dd-4bed-b2f2-8667534d0ceb
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
# ╠═883a4e03-aa64-4b1c-b841-2a7eb73f4771
# ╠═2fc0dae8-bd25-4323-bd8f-d0d27f9eb06d
# ╠═e955f2f6-91c5-476e-9aae-96163de036e4
# ╠═240e4e7b-b90f-4dd8-9862-e3cf4876ab98
# ╠═daa82ef0-fd80-4ece-ba12-93a47a30a176
# ╠═f3007639-619a-4153-bbb2-7699e08a8c1e
# ╠═0d49b778-1df2-47cf-add2-4bebd2ef6bc8
