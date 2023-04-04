### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 8c33d170-55cc-41ef-9a87-f27b644a5133
######### BridgeStan Bernoulli example  ###########

begin
	using BridgeStan
	using DataFrames
	using StanSample
end

# ╔═╡ 5546d52c-a124-4ba7-acd3-9d8b28d78e93
ProjDir = "/Users/rob/.julia/dev/StanExampleNotebooks/notebooks/BridgeStan";

# ╔═╡ 07c1d0b3-d993-4a20-9c8f-a1143527fdea


# ╔═╡ 2b25bf00-406f-4e46-a398-8ab6f61c7969
isfile(joinpath(ProjDir, "bernoulli.stan"))

# ╔═╡ 72a9c5e8-cc16-4d06-be05-73d403448b4c
isfile(joinpath(ProjDir, "bernoulli_data_2.json"))

# ╔═╡ 2c33370f-7b12-4d5b-a48a-880ca33a5ab4
get_bridgestan_path()

# ╔═╡ 98b7a7a1-09e9-4742-84e6-6b3eaf6242f6
smb = BridgeStan.StanModel(;
    stan_file = joinpath(ProjDir, "bernoulli.stan"),
    stanc_args = ["--warn-pedantic --O1"],
    make_args = ["CXX=clang++", "STAN_THREADS=true"],
    data=joinpath(ProjDir, "bernoulli_data_2.json"),
    seed = 204,
    chain_id = 0
)

# ╔═╡ aab341e5-3495-41a6-ba32-c2c47fef8265
println("This model's name is $(BridgeStan.name(smb)).")

# ╔═╡ 8e97fa1c-ddbd-4a4a-b823-1c7eee69e55d
println("It has $(BridgeStan.param_num(smb)) parameters.")

# ╔═╡ 971aed05-21be-4c77-9f8d-85a50af4aa7b
if typeof(smb) == BridgeStan.StanModel
    x = rand(BridgeStan.param_unc_num(smb))
    q = @. log(x / (1 - x))        # unconstrained scale

    lp, grad = BridgeStan.log_density_gradient(smb, q, jacobian = 0)

    "log_density and gradient of Bernoulli model: $(round(lp[1]; digits=3)), $(round(grad[1]; digits=3))"
	
end

# ╔═╡ a006e817-57dc-468f-b051-56e7b94334fd
if typeof(smb) == BridgeStan.StanModel
    function sim(smb::BridgeStan.StanModel, x=LinRange(0.1, 0.9, 100))
        q = zeros(length(x))
        ld = zeros(length(x))
        g = Vector{Vector{Float64}}(undef, length(x))
        for (i, p) in enumerate(x)
            q[i] = @. log(p / (1 - p)) # unconstrained scale
            ld[i], g[i] = BridgeStan.log_density_gradient(smb, q[i:i],
                jacobian = 0)
        end
        return DataFrame(x=x, q=q, log_density=ld, gradient=g)
    end

  sim(smb)

end

# ╔═╡ Cell order:
# ╠═8c33d170-55cc-41ef-9a87-f27b644a5133
# ╠═5546d52c-a124-4ba7-acd3-9d8b28d78e93
# ╠═07c1d0b3-d993-4a20-9c8f-a1143527fdea
# ╠═2b25bf00-406f-4e46-a398-8ab6f61c7969
# ╠═72a9c5e8-cc16-4d06-be05-73d403448b4c
# ╠═2c33370f-7b12-4d5b-a48a-880ca33a5ab4
# ╠═98b7a7a1-09e9-4742-84e6-6b3eaf6242f6
# ╠═aab341e5-3495-41a6-ba32-c2c47fef8265
# ╠═8e97fa1c-ddbd-4a4a-b823-1c7eee69e55d
# ╠═971aed05-21be-4c77-9f8d-85a50af4aa7b
# ╠═a006e817-57dc-468f-b051-56e7b94334fd
