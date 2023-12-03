### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ 1063debc-9e55-417d-9cc0-5b040a1f60af
using Pkg

# ╔═╡ 3dda1456-27a7-43b9-94df-8c1e66902ec1
Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

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
md" ## Why example chapter 1.2.1"

# ╔═╡ 6a8fe376-cf9d-4b39-a601-8f4763f6737b
md"##### Set page layout for notebook."

# ╔═╡ d3ce5ff1-3429-4a3b-93fd-9f246db34fe2
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(5px, 3%);
    	padding-right: max(5px, 15%);
	}
</style>
"""

# ╔═╡ b3df5ef5-f979-44b0-a562-314566c2eed2
begin
	N_drug = 350
	N_no_drug = 350
	
	N_men_drug = 87
	N_men_no_drug = 270

	N_women_drug = 263
	N_women_no_drug = 80
	tbl = DataFrame()
	tbl.sex = [:male, :female, :combined]
	tbl.recover_drug = [[81, 87], [192, 263], [273, 350]]
	tbl.recover_nodrug = [[234, 270], [55, 80], [289, 350]]
	tbl
end

# ╔═╡ e50c8afb-1e30-4bea-be4e-7bedf27819cd
sum(rand(Bernoulli(0.93), 87))

# ╔═╡ 74ae7fcc-21ee-4fdd-82b9-41ea5b433a02
begin
	df = DataFrame()
	df.sex = vcat(repeat([:male], 353), repeat([:female], 347))
	df.drug = vcat(repeat([:true], 350), repeat([:false], 350))
	df.recovered = vcat(
		repeat([:true], 81), repeat([:false], 6), repeat([:true], 192), repeat([:false], 71),
		repeat([:true], 234), repeat([:false],36), repeat([:true], 55), repeat([:false], 25))
	df
end
	

# ╔═╡ 512999b1-4eda-472b-baf7-8f273902b7fa
begin
	p_men_drug = 87/357
	p_recover_men_drug = 0.93
	p_recover_men_no_drug = 0.87

	p_women_drug = 243/343
	p_recover_women_drug = 0.73
	p_recover_women_no_drug = 0.83
end

# ╔═╡ 808c0d3e-6b70-4ff7-b841-cabd9fc66ade
begin
	df1 = DataFrame()
	df1.sex = vcat(repeat([:male], 353), repeat([:female], 347))
	df1.drug = repeat([:false], nrow(df1))
	df1.recovered = repeat([:false], nrow(df1))
	for r in eachrow(df1)
		if r.sex == :Male
			r.drug = rand(Binomial(1, p_men_drug))
			if r.drug == true
				r.recovered = rand(Binomial(1, p_recover_men_drug))
			else
				r.recovered = rand(Binomial(1, p_recover_men_no_drug))
			end
		else
			r.drug = rand(Binomial(1, p_women_drug))
			if r.drug == true
				r.recovered = rand(Binomial(1, p_recover_women_drug))
			else
				r.recovered = rand(Binomial(1, p_recover_women_no_drug))
			end
		end
	end
	df1
end

# ╔═╡ 0f2dbfe9-6d49-4305-b947-2feca8964426
begin
	tbl1 = DataFrame()
	tbl1.sex = [:male, :female, :Combined]
	drugs = df1[df1.drug .== true, :]
	no_drugs = df1[df1.drug .== false, :]
	m_d = drugs[drugs.sex .== :male, :]
	m_nd = no_drugs[no_drugs.sex .== :male, :]
	f_d = drugs[drugs.sex .== :female, :]
	f_nd = no_drugs[no_drugs.sex .== :female, :]
	m_d_r = m_d[m_d.recovered .== true, :]
	m_nd_r = m_nd[m_nd.recovered .== true, :]
	f_d_r = f_d[f_d.recovered .== true, :]
	f_nd_r = f_nd[f_nd.recovered .== true, :]
	m = [[nrow(m_d_r), nrow(m_d)], [nrow(f_d_r), nrow(f_d)],  [nrow(m_d_r)+nrow(f_d_r), nrow(m_d)+nrow(m_d)]]
	f = [[nrow(m_nd_r), nrow(m_nd)], [nrow(f_nd_r), nrow(f_nd)], [nrow(m_nd_r)+nrow(f_nd_r), nrow(f_d)+nrow(f_nd)]]
	tbl1.recover_drug = m
	tbl1.recover_nodrug = f
	tbl1
end

# ╔═╡ 479abf89-8ce6-428e-8397-788242e966b6
f

# ╔═╡ fbd5fce5-6962-450c-8dc6-17d0d8f92a71
versioninfo()

# ╔═╡ Cell order:
# ╟─7d41553e-779f-4cb5-8563-08b8afc4b7ed
# ╟─6a8fe376-cf9d-4b39-a601-8f4763f6737b
# ╠═d3ce5ff1-3429-4a3b-93fd-9f246db34fe2
# ╠═1063debc-9e55-417d-9cc0-5b040a1f60af
# ╠═3dda1456-27a7-43b9-94df-8c1e66902ec1
# ╠═aa89f177-03d8-409e-916e-208b9b897ea4
# ╠═b3df5ef5-f979-44b0-a562-314566c2eed2
# ╠═e50c8afb-1e30-4bea-be4e-7bedf27819cd
# ╠═74ae7fcc-21ee-4fdd-82b9-41ea5b433a02
# ╠═512999b1-4eda-472b-baf7-8f273902b7fa
# ╠═808c0d3e-6b70-4ff7-b841-cabd9fc66ade
# ╠═0f2dbfe9-6d49-4305-b947-2feca8964426
# ╠═479abf89-8ce6-428e-8397-788242e966b6
# ╠═fbd5fce5-6962-450c-8dc6-17d0d8f92a71
