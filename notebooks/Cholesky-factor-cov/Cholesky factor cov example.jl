### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 5084b8f0-65ac-4704-b1fc-2a9008132bd7
using Pkg

# ╔═╡ d7753cf6-7452-421a-a3ec-76e07646f808
Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 550371ad-d411-4e66-9d63-7329322c6ea1
begin
    # Specific to this notebook
    using GLM
    using Statistics
	using Test
	using Distributed: pmap

    # Specific to ROSStanPluto
    using StanSample
    
    # Graphics related
    using CairoMakie
    using AlgebraOfGraphics
    
    # Include basic packages
    using RegressionAndOtherStories
end

# ╔═╡ eb7ea04a-da52-4e69-ac3e-87dc7f014652
md"## Cholesky factor cov example"

# ╔═╡ cf39df58-3371-4535-88e4-f3f6c0404500
md" ###### Widen the cells."

# ╔═╡ 0616ece8-ccf8-4281-bfed-9c1192edf88e
html"""
<style>
    main {
        margin: 0 auto;
        max-width: 2000px;
        padding-left: max(160px, 10%);
        padding-right: max(160px, 15%);
    }
</style>
"""

# ╔═╡ 4755dab0-d228-41d3-934a-56f2863a5652
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ 5df93d41-80d4-419e-b9cd-46ae2b8389cf
stan_data = (N = 3, nu = 13, L_Psi = [1.0 0.0 0.0; 2.0 3.0 0.0; 4.0 5.0 6.0]);

# ╔═╡ 94ed97ce-5997-4fba-b087-6c8be92f0d22
stan_data_2 = Dict("N" => 3, "nu" => 13, "L_Psi" => [1.0 0.0 0.0; 2.0 3.0 0.0; 4.0 5.0 6.0]);

# ╔═╡ 49489ad4-2195-48ff-94b0-bc428f334b49
model_code = "
data {
    int<lower=1> N;
    real<lower=N-1> nu;
    cholesky_factor_cov[N] L_Psi;
}
parameters {
    cholesky_factor_cov[N] L_X;
}
model {
    L_X ~ inv_wishart_cholesky(nu, L_Psi);
}
";

#"/Users/rob/.julia/dev/Stan/test/Examples-Test-Cases/Cholesky-factor-cov"

# ╔═╡ 65a6d377-4fc3-4c6e-82d6-0c1904b9b6d1
cd(joinpath("/Users", "rob", ".julia", "dev", "StanSample", "test", "test_cholesky_factor_cov"))

# ╔═╡ 2dbe3685-47ca-4b3e-93a4-a70f67aac4d4
sm = SampleModel("test", model_code);

# ╔═╡ 1787d3d3-3482-433d-aeb3-c6948d38e97d
rc_2 = stan_sample(sm; data=stan_data_2);

# ╔═╡ 244e832c-b05f-406f-815e-a7af268fffb8
success(rc_2)

# ╔═╡ b7b9053c-3a40-4a63-9e28-1ea9583af7bd
ndf1 = read_samples(sm, :nesteddataframe);

# ╔═╡ 5dd27e46-bff4-43d0-8c39-26393e29347d
ndf1.L_X[1]

# ╔═╡ 2d0b7d8d-3def-43b5-b262-1236ae97d438
for j in 1:3
    rc = stan_sample(sm; data=stan_data)
    ndf2 = read_samples(sm, :nesteddataframe)
    println(ndf2.L_X[1])
end    

# ╔═╡ 2660f495-ed3e-4eaf-a1ae-6f8a69690f33
stan_data_3 = Dict("N" => 3, "nu" => 13, "L_Psi" => ndf1.L_X[1]);

# ╔═╡ 3eedf51c-0d02-4725-b2f1-bcf58fae2cba
 begin
	rc = stan_sample(sm; data=stan_data_3)
    ndf3 = read_samples(sm, :nesteddataframe)
    ndf3.L_X[1]
 end

# ╔═╡ 9d897891-e543-4bc1-b2e9-0fef5a299161
let
	StanBase.update_json_files(sm, stan_data_2, 1, "data")
	read(sm.output_base*"_data_1.json", String)
end

# ╔═╡ c6aa635e-e543-4279-887b-6980c82f3bcc
let
	StanBase.update_json_files(sm, stan_data, 1, "data")
	read(sm.output_base*"_data_1.json", String)
end

# ╔═╡ Cell order:
# ╟─eb7ea04a-da52-4e69-ac3e-87dc7f014652
# ╟─cf39df58-3371-4535-88e4-f3f6c0404500
# ╠═0616ece8-ccf8-4281-bfed-9c1192edf88e
# ╟─4755dab0-d228-41d3-934a-56f2863a5652
# ╠═5084b8f0-65ac-4704-b1fc-2a9008132bd7
# ╠═d7753cf6-7452-421a-a3ec-76e07646f808
# ╠═550371ad-d411-4e66-9d63-7329322c6ea1
# ╠═5df93d41-80d4-419e-b9cd-46ae2b8389cf
# ╠═94ed97ce-5997-4fba-b087-6c8be92f0d22
# ╠═49489ad4-2195-48ff-94b0-bc428f334b49
# ╠═65a6d377-4fc3-4c6e-82d6-0c1904b9b6d1
# ╠═2dbe3685-47ca-4b3e-93a4-a70f67aac4d4
# ╠═1787d3d3-3482-433d-aeb3-c6948d38e97d
# ╠═244e832c-b05f-406f-815e-a7af268fffb8
# ╠═b7b9053c-3a40-4a63-9e28-1ea9583af7bd
# ╠═5dd27e46-bff4-43d0-8c39-26393e29347d
# ╠═2d0b7d8d-3def-43b5-b262-1236ae97d438
# ╠═2660f495-ed3e-4eaf-a1ae-6f8a69690f33
# ╠═3eedf51c-0d02-4725-b2f1-bcf58fae2cba
# ╠═9d897891-e543-4bc1-b2e9-0fef5a299161
# ╠═c6aa635e-e543-4279-887b-6980c82f3bcc
