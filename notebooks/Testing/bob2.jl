### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ cd0534dd-2538-4d18-b52e-c0e61326b52c
using Pkg

# ╔═╡ 0f05ca70-8671-4f11-8bbb-bf8a91eee964
Pkg.activate(expanduser(joinpath("~", ".julia", "dev", "SR2StanPluto")))

# ╔═╡ c2a1ca27-baec-4c11-b4fe-7cd159b3107c
begin
	# Script specific
    using BSplines

	# Graphics related
	using CairoMakie
	
	# Stan related
	using StanSample, StanQuap

	# Project related
	using StatisticalRethinking: sr_datadir, scale!
	using RegressionAndOtherStories
end


# ╔═╡ d675725c-5b02-4333-ba81-fc606c57c43a
md" ##### Widen the cells."

# ╔═╡ c67b0d6a-b2ee-4577-84d3-6544730f02a1
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 20%);
	}
</style>
"""

# ╔═╡ 66acc599-bb05-4954-ae4c-e7b6cdb2df7d
function rw_cov_matrix(d, rho)
    sigma = zeros(d, d)
    for i in 1:d
        for j in 1:d
            sigma[i, j] = rho ^ abs(i - j)
		end
	end
    return sigma
end

# ╔═╡ 9f6c29de-7275-4a85-866f-ffd32600beb6
function random_predictors(n, d, rho)
    sigma = rw_cov_matrix(d, rho)
    mu = zeros(d)
    x = rand(MvNormal(mu, sigma), n)
    return x
end

# ╔═╡ 4d0da332-f940-4bd8-a2ed-deb615c6762f
sq_error(u, v) = sum((u - v).^2)

# ╔═╡ 5512bfb8-7769-4422-a584-f83b396deb6e
inv_logit(u) = 1 / (1 + exp(-u))

# ╔═╡ f203e2ca-18d6-496c-a094-7775e5a03b9e
begin
	D = 3         # number of predictors including intercept
	N = 5         # number of data points used to train
	rho = 0.9     # correlation of predictor RW covariance
end

# ╔═╡ d7f46f8d-48f6-428e-bc5e-48f95f7ecbd7
logstan = "
data {
  int<lower=1> D;                         // num predictors
  int<lower=0> N;                         // num observations
  matrix[N, D] x;                         // covariates
  array[N] int<lower=0, upper=1> y;       // outcomes Y[n] = y[n]
}
parameters {
  vector[D] beta;                         // regression coefficients
}
model {
  y ~ bernoulli_logit(x * beta);     // likelihood: logistic regression
  beta ~ normal(0, 1);               // prior:      ridge
}
";

# ╔═╡ 43692b69-9422-450b-ad13-29eaf65a7f10
begin
	tmpdir = joinpath(@__DIR__, "tmp")
	model_logistic = SampleModel("logistic", logstan, tmpdir)
end;

# ╔═╡ 2ea84f0e-8c5b-4554-b4c5-c3391c61c57e
function fit_bayes_draws(model, data)
	rc = stan_sample(model; data, num_chains=1)
	if success(rc)
		res = read_samples(model, :nesteddataframe)
	end
	return res
end	

# ╔═╡ ccbe6312-77c2-49ee-97c4-3a7f61d5647e
begin
	df = DataFrame(error=[], estimator=[], data=[])
	beta = Vector{Float64}[]
	x = zeros(N, D)
	beta = rand(Normal(0, 1), D)
	println("size(beta) = $(size(beta))")
	x = Matrix(random_predictors(N, D, rho))'
	x[:, 1] .= 1.0
	println("size(x) = $(size(x))")
	e_log_odds = x * beta
	println("size(e_log_odds) = $(size(e_log_odds))")
	e_y = inv_logit.(e_log_odds)
	y_max = map(x -> x > 0.5 ? 1 : 0, e_y)
	println("size(y_max) = $(size(y_max))")
	data = (D=D, N=N, x=Matrix(x), y=y_max)
	beta_draws_max = fit_bayes_draws(model_logistic, data)
end

# ╔═╡ 7d1d0472-725c-4a35-b90c-40fa18c7a015
df_beta = DataFrame(beta_draws_max)

# ╔═╡ 9c3d485f-7cc1-43b3-94cf-1bacf7455274
Matrix(df_beta)

# ╔═╡ 703b73b2-7fd2-49ec-a2ec-74163f3a32c1
data

# ╔═╡ a5403db2-415f-4177-a78f-ded26c771db0
beta

# ╔═╡ cf83b62b-211b-4a50-9f65-190d379545b4
x

# ╔═╡ a4484bf8-b66d-49a0-9d4c-a1bc958e8fd6
[dot(x[i,:], beta) for i in 1:size(x, 1)]

# ╔═╡ a8c44397-ed17-4927-849e-3eb166c66d8a
x * beta

# ╔═╡ Cell order:
# ╠═d675725c-5b02-4333-ba81-fc606c57c43a
# ╠═c67b0d6a-b2ee-4577-84d3-6544730f02a1
# ╠═cd0534dd-2538-4d18-b52e-c0e61326b52c
# ╠═0f05ca70-8671-4f11-8bbb-bf8a91eee964
# ╠═c2a1ca27-baec-4c11-b4fe-7cd159b3107c
# ╠═66acc599-bb05-4954-ae4c-e7b6cdb2df7d
# ╠═9f6c29de-7275-4a85-866f-ffd32600beb6
# ╠═4d0da332-f940-4bd8-a2ed-deb615c6762f
# ╠═5512bfb8-7769-4422-a584-f83b396deb6e
# ╠═f203e2ca-18d6-496c-a094-7775e5a03b9e
# ╠═d7f46f8d-48f6-428e-bc5e-48f95f7ecbd7
# ╠═43692b69-9422-450b-ad13-29eaf65a7f10
# ╠═2ea84f0e-8c5b-4554-b4c5-c3391c61c57e
# ╠═ccbe6312-77c2-49ee-97c4-3a7f61d5647e
# ╠═7d1d0472-725c-4a35-b90c-40fa18c7a015
# ╠═9c3d485f-7cc1-43b3-94cf-1bacf7455274
# ╠═703b73b2-7fd2-49ec-a2ec-74163f3a32c1
# ╠═a5403db2-415f-4177-a78f-ded26c771db0
# ╠═cf83b62b-211b-4a50-9f65-190d379545b4
# ╠═a4484bf8-b66d-49a0-9d4c-a1bc958e8fd6
# ╠═a8c44397-ed17-4927-849e-3eb166c66d8a
