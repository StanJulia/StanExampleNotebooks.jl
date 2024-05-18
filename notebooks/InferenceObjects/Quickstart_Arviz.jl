### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 81df2ce6-4ee3-493f-abca-b774f3c088ee
using Pkg

# ╔═╡ ea1ce099-2e88-411d-a21d-2aa4b9d296e6
Pkg.activate(expanduser("~/.julia/dev/ArviZPlutoExamples"))

# ╔═╡ 4863ebc2-d3bb-4f69-95d9-e663431603c0
begin
	using PSIS
	using Distributions
	using ArviZ
	using ArviZPythonPlots
	using LinearAlgebra
	using Random
	using StanSample
	using Turing
end

# ╔═╡ cbae3d0b-f996-480e-b232-605d967119e8
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""


# ╔═╡ 678cff8e-f16b-4060-bc3c-b2f931056bc7
proposal = Normal()

# ╔═╡ 4cbb43ae-4dd9-40dc-ade6-4e0cd188e17c
target = TDist(7)

# ╔═╡ e225a94b-5e5a-41ac-b1cc-0b493d952cd6
let
	ndraws, nchains, nparams = (1_000, 1, 30)
	x = rand(proposal, ndraws, nchains, nparams)
	log_ratios = logpdf.(target, x) .- logpdf.(proposal, x)
	result = psis(log_ratios)
end

# ╔═╡ 08f7b89d-b2d8-45c9-80cc-207ff1b5164f
rng1 = Random.MersenneTwister(37772);

# ╔═╡ b0510f73-e92b-4a22-8458-63de45a5a23a
begin
    plot_posterior(randn(rng1, 100_000))
    gcf()
end

# ╔═╡ 849e4cdc-da91-4cd5-8d73-43bc2d9e7065
let
    s = (50, 10)
    plot_forest((
        normal=randn(rng1, s),
        gumbel=rand(rng1, Gumbel(), s),
        student_t=rand(rng1, TDist(6), s),
        exponential=rand(rng1, Exponential(), s),
    ),)
    gcf()
end

# ╔═╡ 8521309f-0b63-4348-9fa9-029424510f94
begin
    J = 8
    y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    schools = [
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "Phillips Exeter",
        "Hotchkiss",
        "Lawrenceville",
        "St. Paul's",
        "Mt. Hermon",
    ]
    ndraws = 1_000
    ndraws_warmup = 1_000
    nchains = 4
end;

# ╔═╡ 59f02339-34e8-4763-98e0-2a2ecbe1d62b
Turing.@model function model_turing(y, σ, J=length(y))
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ ~ filldist(Normal(μ, τ), J)
    for i in 1:J
        y[i] ~ Normal(θ[i], σ[i])
    end
end

# ╔═╡ 36b25e30-ffe5-4903-81ad-fdee51a2163d
rng2 = Random.MersenneTwister(16653);

# ╔═╡ b480e383-a181-4441-ac94-65ac64bdd90f

begin
    param_mod_turing = model_turing(y, σ)
    sampler = NUTS(ndraws_warmup, 0.8)

	# chns10_3t = mapreduce(c -> sample(m10_3t, sampler, nsamples), chainscat, 1:nchains)

    turing_chns = mapreduce(c -> Turing.sample(
        model_turing(y, σ), sampler, ndraws), chainscat, 1:4
    )
end;


# ╔═╡ 46f04c5f-7064-4e1b-875a-b401e3f30392
let
    plot_autocorr(turing_chns; var_names=(:μ, :τ))
    gcf()
end

# ╔═╡ da3a0a24-fb7c-478a-895a-689338e4c3cc
idata_turing_post = from_mcmcchains(
    turing_chns;
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
    library="Turing",
)

# ╔═╡ 439417b3-05be-473f-b698-8273de206875
idata_turing_post.posterior

# ╔═╡ 0d874ea4-ae09-4fae-924d-21cd2abe2309
begin
    plot_trace(idata_turing_post)
    gcf()
end

# ╔═╡ 744302c4-418d-4e34-98fc-40e18bc6163f
begin
    plot_energy(idata_turing_post)
    gcf()
end

# ╔═╡ 37423772-a220-428a-9e0d-92b997b651da
prior = mapreduce(c->Turing.sample(rng2, param_mod_turing, Prior(), ndraws), chainscat, 1:4);

# ╔═╡ 6ad92a23-4ea4-4c48-8287-0141daeb5f6d
begin
    # Instantiate the predictive model
    param_mod_predict = model_turing(similar(y, Missing), σ)
    # and then sample!
    prior_predictive = Turing.predict(rng2, param_mod_predict, prior)
    posterior_predictive = Turing.predict(rng2, param_mod_predict, turing_chns)
end;

# ╔═╡ 123769a8-e128-4283-8f21-a22ecfcea0e7
log_likelihood = let
    log_likelihood = Turing.pointwise_loglikelihoods(
        param_mod_turing, MCMCChains.get_sections(turing_chns, :parameters)
    )
    # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
    ynames = string.(keys(posterior_predictive))
    log_likelihood_y = getindex.(Ref(log_likelihood), ynames)
    (; y=cat(log_likelihood_y...; dims=3))
end;

# ╔═╡ 9457341b-5198-4f56-94cb-c0fbb4e87f54
idata_turing = from_mcmcchains(
    turing_chns;
    posterior_predictive,
    log_likelihood,
    prior,
    prior_predictive,
    observed_data=(; y),
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
    library=Turing,
)

# ╔═╡ b4164215-15bb-4dd6-b312-aec0d383b83c
loo(idata_turing) # higher ELPD is better

# ╔═╡ 6a892b20-b30d-47f4-8e46-81a51d82dc1c
begin
    plot_loo_pit(idata_turing; y=:y, ecdf=true)
    gcf()
end

# ╔═╡ 9d4dbf99-6ddf-4b12-8af0-60f369cc7ea3
begin
    schools_code = """
    data {
      int<lower=0> J;
      array[J] real y;
      array[J] real<lower=0> sigma;
    }

    parameters {
      real mu;
      real<lower=0> tau;
      array[J] real theta;
    }

    model {
      mu ~ normal(0, 5);
      tau ~ cauchy(0, 5);
      theta ~ normal(mu, tau);
      y ~ normal(theta, sigma);
    }

    generated quantities {
        vector[J] log_lik;
        vector[J] y_hat;
        for (j in 1:J) {
            log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
            y_hat[j] = normal_rng(theta[j], sigma[j]);
        }
    }
    """

    schools_data = Dict("J" => J, "y" => y, "sigma" => σ)
    idata_stan = mktempdir() do path
        stan_model = SampleModel("schools", schools_code, path)
        _ = stan_sample(
            stan_model;
            data=schools_data,
            num_chains=nchains,
            num_warmups=ndraws_warmup,
            num_samples=ndraws,
            seed=28983,
            summary=false,
        )
        return StanSample.inferencedata(
            stan_model;
            posterior_predictive_var=:y_hat,
            observed_data=(; y),
            log_likelihood_var=:log_lik,
            coords=(; school=schools),
            dims=NamedTuple(
                k => (:school,) for k in (:y, :sigma, :theta, :log_lik, :y_hat)
            ),
        )
    end
end

# ╔═╡ 80b3e428-fdfb-4989-8254-117cc7673cf5
idata_stan

# ╔═╡ bdbb6f4a-3fd4-47b0-aeed-4b20ffc5b519
begin
    plot_density(idata_stan; var_names=(:mu, :tau))
    gcf()
end

# ╔═╡ 4bdaccad-d8cc-4595-be8e-5cf612fac8dc
begin
    plot_pair(
        idata_stan;
        coords=Dict(:school => ["Choate", "Deerfield", "Phillips Andover"]),
        divergences=true,
    )
    gcf()
end

# ╔═╡ Cell order:
# ╠═cbae3d0b-f996-480e-b232-605d967119e8
# ╠═81df2ce6-4ee3-493f-abca-b774f3c088ee
# ╠═ea1ce099-2e88-411d-a21d-2aa4b9d296e6
# ╠═4863ebc2-d3bb-4f69-95d9-e663431603c0
# ╠═678cff8e-f16b-4060-bc3c-b2f931056bc7
# ╠═4cbb43ae-4dd9-40dc-ade6-4e0cd188e17c
# ╠═e225a94b-5e5a-41ac-b1cc-0b493d952cd6
# ╠═08f7b89d-b2d8-45c9-80cc-207ff1b5164f
# ╠═b0510f73-e92b-4a22-8458-63de45a5a23a
# ╠═849e4cdc-da91-4cd5-8d73-43bc2d9e7065
# ╠═8521309f-0b63-4348-9fa9-029424510f94
# ╠═59f02339-34e8-4763-98e0-2a2ecbe1d62b
# ╠═36b25e30-ffe5-4903-81ad-fdee51a2163d
# ╠═b480e383-a181-4441-ac94-65ac64bdd90f
# ╠═46f04c5f-7064-4e1b-875a-b401e3f30392
# ╠═da3a0a24-fb7c-478a-895a-689338e4c3cc
# ╠═439417b3-05be-473f-b698-8273de206875
# ╠═0d874ea4-ae09-4fae-924d-21cd2abe2309
# ╠═744302c4-418d-4e34-98fc-40e18bc6163f
# ╠═37423772-a220-428a-9e0d-92b997b651da
# ╠═6ad92a23-4ea4-4c48-8287-0141daeb5f6d
# ╠═123769a8-e128-4283-8f21-a22ecfcea0e7
# ╠═9457341b-5198-4f56-94cb-c0fbb4e87f54
# ╠═b4164215-15bb-4dd6-b312-aec0d383b83c
# ╠═6a892b20-b30d-47f4-8e46-81a51d82dc1c
# ╠═9d4dbf99-6ddf-4b12-8af0-60f369cc7ea3
# ╠═80b3e428-fdfb-4989-8254-117cc7673cf5
# ╠═bdbb6f4a-3fd4-47b0-aeed-4b20ffc5b519
# ╠═4bdaccad-d8cc-4595-be8e-5cf612fac8dc
