### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f5ed29c5-7f99-4dd1-bfdc-34f4001b9c34
using Pkg

# ╔═╡ fed755ad-e816-4659-9389-7ef4008c6456
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 283d88d1-0c3e-443e-bf44-63f03e869c12
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

# ╔═╡ 7e562c21-86e0-4b22-8a2c-d0188c7a1ab8
md" ## Counterfactual simulation."

# ╔═╡ 18b36ec3-4a56-49d9-b0f2-c1d266adffb4
md"##### Set page layout for notebook."

# ╔═╡ 41619ae3-be1c-453f-9635-46be85e1e728
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

# ╔═╡ dcd24508-f4df-4b27-acb2-780f679f684d


# ╔═╡ 98d7c715-1fa0-434a-9498-12c54224647d
begin
	waffles = CSV.read(sr_datadir("WaffleDivorce.csv"), DataFrame; delim=';');
	scale_df_cols!(waffles, [:Marriage, :MedianAgeMarriage, :Divorce])
	waffles.Whpm = waffles.WaffleHouses./waffles.Population
	waffles[:, [:Loc, :Population, :Marriage, :MedianAgeMarriage, :Divorce, :Divorce_s]]
end

# ╔═╡ e1eea04a-abdf-49ad-a4ef-e645fbd06a09
let
	A = waffles.MedianAgeMarriage_s
	M = waffles.Marriage_s
	D = waffles.Divorce_s
	global p = 0.01
	global X = [A M D]
	global dfAMD = DataFrame(A=A, M=M, D=D)
end;

# ╔═╡ 9bf77cd3-31bc-49a1-8ab9-3e03071dadf4
dag = create_fci_dag("AMD", dfAMD, "Digraph AMD {A->M; A->D; M->D;}");

# ╔═╡ 9fbe5dbb-793d-4ac7-8227-e86b5fc557b5
gvplot(dag)

# ╔═╡ 2cc487d9-4659-4c07-ac37-b94fbebec8df
stan5_3_A = "
data {
	int N;
	vector[N] D;
	vector[N] M;
	vector[N] A;
}
parameters {
	// A->D + A->M->D
	real a;
	real bA;
	real bM;
	real<lower=0> sigma;
	// A-> M
	real aM;
	real bAM;
	real<lower=0> sigma_M;
}
model {
	// A -> M
	vector[N] mu_M = aM + bAM * A;
	aM ~ normal( 0 , 0.2 );
	bAM ~ normal( 0 , 0.5 );
	sigma_M ~ exponential( 1 );
	M ~ normal( mu_M , sigma_M );
	// A -> M -> D
	vector[N] mu = a + bA * A + bM * M;
	a ~ normal( 0 , 0.2 );
	bA ~ normal( 0 , 0.5 );
	bM ~ normal( 0 , 0.5 );
	sigma ~ exponential( 1 );
	D ~ normal( mu , sigma );
}
";

# ╔═╡ cf696155-bb74-47ec-94dc-99a8563bc270
begin
	data = (N = size(dfAMD, 1), D = waffles.Divorce_s, M = waffles.Marriage_s,
		A = waffles.MedianAgeMarriage_s)
	global m5_3_As = SampleModel("m5.3_A", stan5_3_A)
	global rc5_3_As = stan_sample(m5_3_As; data)
	success(rc5_3_As) && describe(m5_3_As, [:a, :bA, :bM, :sigma, :aM, :bAM, :sigma_M])
end

# ╔═╡ 417e778a-74ce-4aa6-8cb4-e9428034d2b7
if success(rc5_3_As)
	post5_3_As_df = read_samples(m5_3_As, :dataframe)
	ms5_3_As = model_summary(post5_3_As_df, [:a, :bA, :bM, :sigma, :aM, :bAM, :sigma_M])
end

# ╔═╡ 64f1374c-dc92-4149-b81c-edc3d3ece8e3
function simulate2(df, coefs, var_seq, coefs_ext)
  m_sim = simulate2(df, coefs, var_seq)
  d_sim = zeros(size(df, 1), length(var_seq));
  for j in 1:size(df, 1)
    for i in 1:length(var_seq)
      d = Normal(df[j, coefs[1]] + df[j, coefs[2]] * var_seq[i] +
        df[j, coefs_ext[1]] * m_sim[j, i], df[j, coefs_ext[2]])
      d_sim[j, i] = rand(d)
    end
  end
  (m_sim, d_sim)
end

# ╔═╡ 09737625-e032-43ca-9aaa-943428ddf957
function simulate2(df, coefs, var_seq)
  m_sim = zeros(size(df, 1), length(var_seq));
  for j in 1:size(df, 1)
    for i in 1:length(var_seq)
      d = Normal(df[j, coefs[1]] + df[j, coefs[2]] * var_seq[i], df[j, coefs[3]])
      m_sim[j, i] = rand(d)
    end
  end
  m_sim
end

# ╔═╡ ae9bf2c0-6a64-46e5-b761-58030c32fbad
a_seq = range(-2, stop=2, length=30);

# ╔═╡ a5d3fd82-62d2-4739-9fdd-9a746c3f3f99
m_sim, d_sim = simulate(post5_3_As_df, [:aM, :bAM, :sigma_M], a_seq, [:bM, :sigma]);

# ╔═╡ db5d1226-09e1-4022-bf7a-68279cb04fb6
let
	df = post5_3_As_df
	coefs = [:aM, :bAM, :sigma_M]
	coefs_ext = [:bM, :sigma]
	var_seq = -2:1:2
	m1 = [df[j, coefs[1]] + df[j, coefs[2]] * var_seq[i] for j in 1:4000, i in 1:5]
	m2 = [df[j, coefs_ext[1]] * m_sim[j, i] for j in 1:4000, i in 1:5]
	[mean(m1; dims=1); mean(m2; dims=1); mean(m1 + m2; dims=1)]
end

# ╔═╡ 3ac23813-430d-42cc-9efd-3ec5d10bfe37
let
	df = post5_3_As_df
	coefs = [:aM, :bAM, :sigma_M]
	coefs_ext = [:bM, :sigma]
	m1 = [df[j, coefs[1]] + df[j, coefs[2]] * a_seq[i] for j in 1:4000, i in 1:30]
	m2 = [df[j, coefs_ext[1]] * m_sim[j, i] for j in 1:4000, i in 1:30]
	mat = [mean(m1; dims=1); mean(m2; dims=1); mean(m1 + m2; dims=1)]
	f = Figure(resulution=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="Manipulated A", ylabel="Counterfactual D",
		title="Total counterfactual effect of A on D")
	scatter!(a_seq, mat[1,:])
	scatter!(a_seq, mat[2,:])
	scatter!(a_seq, mat[3,:])
	f
end

# ╔═╡ 5f972976-d4e9-4965-b133-fdab819d6d71
let
	f = Figure(resulution=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="Manipulated A", ylabel="Counterfactual D",
		title="Total counterfactual effect of A on D", yticks=-2:1:2)
	
	m, l, u = estimparam(d_sim)
	lines!(a_seq, m)
	band!(a_seq, l, u; color=(:grey, 0.3))

	ax = Axis(f[1, 2]; xlabel="Manipulated A", ylabel="Counterfactual M",
		title="Counterfactual effect of A on M")
	m, l, u = estimparam(m_sim)
	lines!(a_seq, m)
	band!(a_seq, l, u; color=(:grey, 0.3))
	
	f
end

# ╔═╡ ed691d57-bb83-4f32-a2ab-f96407bdb503
md"##### M -> D"

# ╔═╡ abd9a0cc-7525-4b7d-bcb8-a08c99ed59ec
let
	m_seq = range(-2, stop=2, length=30)

	f = Figure(resolution=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="Manipulated A", ylabel="Counterfactual D",
		title="Total counterfactual effect of A on D", yticks=-2:1:2)
	m, l, u = estimparam(d_sim)
	lines!(a_seq, m)
	band!(a_seq, l, u; color=(:grey, 0.3))
	
	md_sim = zeros(size(post5_3_As_df, 1), length(m_seq))
	for j in 1:size(post5_3_As_df, 1)
		for i in 1:length(m_seq)
			d = Normal(post5_3_As_df[j, :a] + post5_3_As_df[j, :bM] * m_seq[i],
				post5_3_As_df[j, :sigma])
			md_sim[j, i] = rand(d, 1)[1]
		end
	end
	ax = Axis(f[1, 2]; xlabel="Manipulated M", ylabel="Counterfactual D",
		title="Counterfactual effect of M on D", yticks=-2:1:2)
	m, l, u = estimparam(md_sim)
	lines!(a_seq, m)
	band!(a_seq, l, u; color=(:grey, 0.3))

	f
end

# ╔═╡ Cell order:
# ╟─7e562c21-86e0-4b22-8a2c-d0188c7a1ab8
# ╟─18b36ec3-4a56-49d9-b0f2-c1d266adffb4
# ╠═41619ae3-be1c-453f-9635-46be85e1e728
# ╠═f5ed29c5-7f99-4dd1-bfdc-34f4001b9c34
# ╠═fed755ad-e816-4659-9389-7ef4008c6456
# ╠═283d88d1-0c3e-443e-bf44-63f03e869c12
# ╠═dcd24508-f4df-4b27-acb2-780f679f684d
# ╠═98d7c715-1fa0-434a-9498-12c54224647d
# ╠═e1eea04a-abdf-49ad-a4ef-e645fbd06a09
# ╠═9bf77cd3-31bc-49a1-8ab9-3e03071dadf4
# ╠═9fbe5dbb-793d-4ac7-8227-e86b5fc557b5
# ╠═2cc487d9-4659-4c07-ac37-b94fbebec8df
# ╠═cf696155-bb74-47ec-94dc-99a8563bc270
# ╠═417e778a-74ce-4aa6-8cb4-e9428034d2b7
# ╠═64f1374c-dc92-4149-b81c-edc3d3ece8e3
# ╠═09737625-e032-43ca-9aaa-943428ddf957
# ╠═ae9bf2c0-6a64-46e5-b761-58030c32fbad
# ╠═a5d3fd82-62d2-4739-9fdd-9a746c3f3f99
# ╠═db5d1226-09e1-4022-bf7a-68279cb04fb6
# ╠═3ac23813-430d-42cc-9efd-3ec5d10bfe37
# ╠═5f972976-d4e9-4965-b133-fdab819d6d71
# ╠═ed691d57-bb83-4f32-a2ab-f96407bdb503
# ╠═abd9a0cc-7525-4b7d-bcb8-a08c99ed59ec
