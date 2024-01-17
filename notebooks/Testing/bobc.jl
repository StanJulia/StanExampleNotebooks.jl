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

# ╔═╡ 2e228dfa-0b36-4363-bea5-ed252f1aae0c
probstan = "
data {
  int<lower=1> D;                        // num predictors
  int<lower=0> N;                        // num observations
  matrix[N, D] x;                        // covariates
  vector<lower=0, upper=1>[N] p;         // Pr[Y[n] = 1]
}
parameters {
  vector[D] beta;                         // regression coefficients
}
model {
  vector[N] E_Y = inv_logit(x * beta);    // expected Y 
  target += sum(p .* log(E_Y));           // likelihood: weighted logistic regression
  target += sum((1 - p) .* log1m(E_Y));   // 
  beta ~ normal(0, 1);                    // prior:    ridge
}
";

# ╔═╡ 30e763fc-7910-4c1a-a3f8-cb245b70b532
oddstan = "
data {
  int<lower=1> D;                              // num predictors
  int<lower=0> N;                              // num observations
  matrix[N, D] x;                              // predictors
  vector<lower=0, upper=1>[N] p;               // Pr[Y_n = 1 | x_n]
}
parameters {
  vector[D] beta;                     // regression coefficients
}
model {
  logit(p) ~ normal(x * beta, 1);     // likelihood: linear regression on log odds
  beta ~ normal(0, 1);                // prior:      ridge
}
";

# ╔═╡ 45a2f2a0-a5b1-410a-b017-1e4840061836
predstan = "
data {
  int<lower=1> D;                                   // num covariates per item
  int<lower=0> N;                                   // num observations
  matrix[N, D] x;                                   // test covariates
  array[N] int<lower=0, upper=1> y;                 // outcomes
}
parameters {
  vector[D] beta;                                   // parameters
}
generated quantities {
  real log_p = bernoulli_logit_lpmf(y | x * beta);  // likelihood
}
";

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
	D = 32         # number of predictors including intercept
	N = 1024         # number of data points used to train
	rho = 0.9     # correlation of predictor RW covariance
	N_test = 5    # number of test items
	M = 32         # number of simulation runs
end

# ╔═╡ 43692b69-9422-450b-ad13-29eaf65a7f10
begin
	tmpdir = joinpath(@__DIR__, "tmp")
	model_logistic = SampleModel("logistic", logstan, tmpdir)
	model_weighted_logistic = SampleModel("weighted_logistic", probstan, tmpdir)
	model_log_odds_linear = SampleModel("odd", oddstan, tmpdir)
	model_predict = SampleModel("pred", predstan, tmpdir)
end;

# ╔═╡ 2ea84f0e-8c5b-4554-b4c5-c3391c61c57e
function fit_bayes_draws(model, data)
	rc = stan_sample(model; data, num_chains=1)
	if success(rc)
		res = read_samples(model, :dataframe)
	end
	return res
end	

# ╔═╡ ccbe6312-77c2-49ee-97c4-3a7f61d5647e
begin
	df = DataFrame()
	for i in 1:M
		beta = rand(Normal(0, 1), D)
		x = Matrix(random_predictors(N, D, rho)')
		x[:, 1] .= 1.0
		e_log_odds = x * beta
		e_y = inv_logit.(e_log_odds)
		y_max = map(x -> x > 0.5 ? 1 : 0, e_y)
		k = length(e_y)
		y_random = [rand(Binomial(1, e_y[k]), 1)[1] for i in 1:k]
		p = e_y
		y_noisy_log_odds = e_log_odds + rand(Normal(0, 1), k)
		p_noisy = inv_logit.(y_noisy_log_odds)
		data_max = (D=D, N=N, x=x, y=y_max)
		data_random = (D=D, N=N, x=x, y=y_random)
		data_probs = (D=D, N=N, x=x, p=p)
		data_noisy_weights = (D=D, N=N, x=x, p=p_noisy)
		global beta_draws_max = fit_bayes_draws(model_logistic, data_max)
    	global beta_draws_random = fit_bayes_draws(model_logistic, data_random)
    	global beta_draws_probs = fit_bayes_draws(model_weighted_logistic, data_probs)
    	global beta_draws_weights = fit_bayes_draws(model_log_odds_linear, data_probs)
    	global beta_draws_noisy_weights = fit_bayes_draws(model_log_odds_linear, data_noisy_weights)
		

		append!(df, DataFrame(
			method="Bayes",
			data="max_prob",
			est=[mean(Array(beta_draws_max); dims=1)[1,:]],
			std=[std(Array(beta_draws_max); dims=1)[1,:]]))
	end
	df
end

# ╔═╡ be3ec458-9bca-42d0-ba0c-43b7f1249e0d
model_summary(beta_draws_max, ["beta.$i" for i in 1:size(beta_draws_max, 2)])

# ╔═╡ 10c50d77-cd65-4ecc-a906-3b2bcee6e325
df

# ╔═╡ Cell order:
# ╠═d675725c-5b02-4333-ba81-fc606c57c43a
# ╠═c67b0d6a-b2ee-4577-84d3-6544730f02a1
# ╠═cd0534dd-2538-4d18-b52e-c0e61326b52c
# ╠═0f05ca70-8671-4f11-8bbb-bf8a91eee964
# ╠═c2a1ca27-baec-4c11-b4fe-7cd159b3107c
# ╠═d7f46f8d-48f6-428e-bc5e-48f95f7ecbd7
# ╠═2e228dfa-0b36-4363-bea5-ed252f1aae0c
# ╠═30e763fc-7910-4c1a-a3f8-cb245b70b532
# ╠═45a2f2a0-a5b1-410a-b017-1e4840061836
# ╠═66acc599-bb05-4954-ae4c-e7b6cdb2df7d
# ╠═9f6c29de-7275-4a85-866f-ffd32600beb6
# ╠═4d0da332-f940-4bd8-a2ed-deb615c6762f
# ╠═5512bfb8-7769-4422-a584-f83b396deb6e
# ╠═f203e2ca-18d6-496c-a094-7775e5a03b9e
# ╠═43692b69-9422-450b-ad13-29eaf65a7f10
# ╠═2ea84f0e-8c5b-4554-b4c5-c3391c61c57e
# ╠═ccbe6312-77c2-49ee-97c4-3a7f61d5647e
# ╠═be3ec458-9bca-42d0-ba0c-43b7f1249e0d
# ╠═10c50d77-cd65-4ecc-a906-3b2bcee6e325
