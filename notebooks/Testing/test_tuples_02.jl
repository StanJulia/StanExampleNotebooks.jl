### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 4530e47d-abe2-4521-ba6f-1e2e4a46cf3a
using Pkg

# ╔═╡ 891015c3-8539-45e1-9a9a-71acfed9cfdf
begin
	using StanSample
	using DataFrames
	using JSON
	using InferenceObjects
	using NamedTupleTools
end

# ╔═╡ 86e386a0-b56f-42f1-a6de-1f15425d1a59
md" ##### Widen the cells."

# ╔═╡ c706075a-0174-450d-a1b0-b202cee4d216
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

# ╔═╡ 79dca9f3-c2d8-430c-b0d5-658638dcf901
stan = "
parameters {
    real r;
    matrix[2, 3] x;
    tuple(real, real) bar;
    tuple(real, tuple(real, real), real) bar2;
    tuple(real, tuple(real, tuple(real, real)), real) bar3;
}
model {
    r ~ std_normal();

    for (i in 1:2) {
        x[i,:] ~ std_normal();
    }

    bar.1 ~ std_normal();
    bar.2 ~ std_normal();
    bar2.1 ~ std_normal();
    bar2.2.1 ~ std_normal();
    bar2.2.2 ~ std_normal();
	bar2.3 ~ std_normal();
    bar3.1 ~ std_normal();
    bar3.2.1 ~ std_normal();
    bar3.2.2.1 ~ std_normal();
    bar3.2.2.2 ~ std_normal();
	bar3.3 ~ std_normal();
}
";

# ╔═╡ bf493f07-d0da-4097-853b-78c9449e4539
begin
	tmpdir = joinpath(@__DIR__, "tmp")
	sm = SampleModel("tuple_model", stan, tmpdir)
	rc = stan_sample(sm)
end;

# ╔═╡ c11b5f61-be28-48f0-9da8-8b8336222dc4
tmpdir

# ╔═╡ d0e4265b-880a-432c-bfc5-04a2a0b7c8c0
chns, col_names = read_samples(sm, :array; return_parameters=true);

# ╔═╡ 037946cf-4622-4b8c-8479-d44451ccbfb1
col_names

# ╔═╡ 0cd5ee24-b6a5-405e-8c83-d731db58472a
size(chns)

# ╔═╡ f72a0bb0-aa13-4b19-9755-3a30604aa2c0
ex1 = StanSample.extract(chns, col_names; permute_dims=true);

# ╔═╡ 267a7b22-6238-40b1-877f-87e1928d96d8
size(ex1[:bar])

# ╔═╡ 9b4d21c6-b16d-40a3-87d3-7131613a7a28
begin
	n = (:bar, )
	v = ((1.0, (2.0, 3.0)), )
	nt = namedtuple(n, v)
end

# ╔═╡ 6e4c6e9f-22d5-4ba4-9e8e-43cb46a3d7fe
function extract(chns::Array{Float64,3}, cnames::Vector{String}; permute_dims=false)
    draws, vars, chains = size(chns)

    ex_dict = Dict{Symbol, Array}()

    group_map = Dict{Symbol, Array}()
    
    for (i, cname) in enumerate(cnames)
        if isnothing(findfirst('.', cname)) && isnothing(findfirst(':', cname))
            ex_dict[Symbol(cname)] = chns[:,i,:]
        elseif !isnothing(findfirst('.', cname))
            sp_arr = split(cname, ".")
            name = Symbol(sp_arr[1])
            if !(name in keys(group_map))
                group_map[name] = Any[]
            end
            push!(group_map[name], (i, [Meta.parse(i) for i in sp_arr[2:end]]))
        elseif !isnothing(findfirst(':', cname))
            sp_arr = split(cname, ":")
            name = Symbol(sp_arr[1])
            if !(name in keys(group_map))
                group_map[name] = Any[]
            end
            push!(group_map[name], (i, [Meta.parse(i) for i in sp_arr[2:end]]))
        end
    end

    println()
    println(group_map)
    println()

    for  (name, group) in group_map
        if !isnothing(findfirst('.', cnames[group[1][1]]))
            max_idx = maximum(hcat([idx for (i, idx) in group_map[name]]...), dims=2)[:,1]
            ex_dict[name] = similar(chns, max_idx..., draws, chains)
            for (j, idx) in group_map[name]
                ex_dict[name][idx..., :, :] = chns[:,j,:]
            end
        elseif !isnothing(findfirst(':', cnames[group[1][1]]))
 			indx_arr = Int[]
			for (j, idx) in group_map[name]
				append!(indx_arr, j)
			end
			max_idx2 = [1, length(indx_arr)]
			ex_dict[name] = similar(chns, max_idx2..., draws, chains)
			#println(size(ex_dict[name]))
			cnt = 0
			for (j, idx) in group_map[name]
				cnt += 1
				#println([j, idx, cnt])
				ex_dict[name][1, cnt, :, :] = chns[:,j,:]
			end
        end
    end

    if permute_dims
        for key in keys(ex_dict)
            if length(size(ex_dict[key])) > 2
                tmp = 1:length(size(ex_dict[key]))
                perm = (tmp[end-1], tmp[end], tmp[1:end-2]...)
                ex_dict[key] = permutedims(ex_dict[key], perm)
            end
        end
    end


    for name in keys(ex_dict)
        if name in [:treedepth__, :n_leapfrog__]
            ex_dict[name] = convert(Matrix{Int}, ex_dict[name])
        elseif name == :divergent__
            ex_dict[name] = convert(Matrix{Bool}, ex_dict[name])
        end
    end

    return (;ex_dict...)
end


# ╔═╡ 965a04e5-bbd8-4184-b30c-f32b5ddcbeee
ex2 = extract(chns, col_names; permute_dims=true);

# ╔═╡ 79643384-8013-41f9-bf32-4f0432ba139a
keys(ex2)

# ╔═╡ d7be571d-9348-4720-854e-9f4342de6f01
size(ex2.bar3)

# ╔═╡ 6f1a1f67-38c0-4c13-bd2c-dd061e4116ac
ex2.bar3[1, 1, :, :]

# ╔═╡ 824db4b1-580e-4dfd-ac1a-c628acda6909
df = read_samples(sm, :dataframe)[:, 14:18]

# ╔═╡ df11a159-d82a-4c3a-93ee-a5a5a5e64fc4
n2 = names(df)

# ╔═╡ e6456796-0616-4911-a89f-294684596d70
gm = Dict{Symbol, Array}(:bar => Any[(8, [1]), (9, [2])], :bar2 => Any[(10, [1]), (11, [2, 1]), (12, [2, 2]), (13, [3])], :bar3 => Any[(14, [1]), (15, [2, 1]), (16, [2, 2, 1]), (17, [2, 2, 2]), (18, [3])], :x => Any[(2, [1, 1]), (3, [2, 1]), (4, [1, 2]), (5, [2, 2]), (6, [1, 3]), (7, [2, 3])]);

# ╔═╡ 72ab49ea-84af-4371-90b5-9d0c20803db1
display(gm)

# ╔═╡ 1af6da5f-2444-47c2-8a2c-d4f79b91e2a7
[idx for (i, idx) in gm[:x]]

# ╔═╡ 33c784cb-1e1d-4db5-9262-7e8efdb258b8
hcat([idx for (i, idx) in gm[:x]]...)

# ╔═╡ a269c980-b52d-4e60-9fda-bfb6a16620b1
maximum(hcat([idx for (i, idx) in gm[:x]]...), dims=2)

# ╔═╡ 744b4729-29ee-4f53-8ba8-c8a8500179ed
maximum(hcat([idx for (i, idx) in gm[:x]]...), dims=2)[:, 1]

# ╔═╡ 0802de94-e5a1-46f3-b0a6-259d51e7f4ae
max_idx = maximum(hcat([idx for (i, idx) in gm[:x]]...), dims=2)[:,1]

# ╔═╡ 87379808-c6cd-4693-aceb-31c9d9ced30d
begin
	draws, vars, chains = size(chns)
	size(similar(chns, max_idx..., draws, chains))
end

# ╔═╡ 3fadbfa4-2076-4c9b-9e43-4fe0da0d08ce
begin
	ex3 = Dict{Symbol, Array}()
	ex3[:x] = similar(chns, max_idx..., draws, chains)
	size(ex3[:x])
end

# ╔═╡ 7157a231-7a49-4a99-b597-a378e61c28cf
begin
	for (j, idx) in gm[:x]
		ex3[:x][idx..., :, :] = chns[:,j,:]
	end
	ex3[:x][:, :, 1, 1]
end

# ╔═╡ 2bf45387-003a-47e4-9669-5e42929b9523
[idx for (i, idx) in gm[:bar]]

# ╔═╡ a5257168-d4c3-4125-9f57-0dcd1ae38789
[idx for (i, idx) in gm[:bar2]]

# ╔═╡ 4d5beb0f-fb84-46ca-ade8-cd83421ee1e6
[idx for (i, idx) in gm[:bar3]]

# ╔═╡ 408389bd-7126-499c-ac94-62473643573a
begin
	indx_arr = Int[]
	for (j, idx) in gm[:bar]
		append!(indx_arr, j)
	end
	max_idx2 = [1, length(indx_arr)]
	ex3[:bar] = similar(chns, max_idx2..., draws, chains)
	size(ex3[:bar])
end

# ╔═╡ 0f3658c2-c417-4f45-8721-1b0fd767b1ca
indx_arr

# ╔═╡ da901259-1ac6-44b4-92e3-43b024b95e65
max_idx2

# ╔═╡ 055ff913-e461-4027-a5c6-059941de8dcf
begin
	cnt = 0
	for (j, idx) in gm[:bar]
		cnt += 1
		ex3[:bar][1, cnt, :, :] = chns[:,j,:]
	end
	size(ex3[:bar])
end

# ╔═╡ 06502f74-7a3b-4c55-8e51-8211ee574598
ex3[:bar][:, :, 1, 1]

# ╔═╡ c7eef545-e660-42de-8e8f-d382e89fd592
chns[1, 8:9, 1]

# ╔═╡ def76ee3-b8fb-46a0-9f37-262e914cd8d2
let
	indx_arr = Int[]
	for (j, idx) in gm[:bar2]
		append!(indx_arr, j)
	end
	max_idx2 = [1, length(indx_arr)]
	ex3[:bar2] = similar(chns, max_idx2..., draws, chains)
	cnt = 0
	for (j, idx) in gm[:bar2]
		cnt += 1
		println([j, idx, cnt])
		ex3[:bar2][1, cnt, :, :] = chns[:,j,:]
	end
	size(ex3[:bar2])
end

# ╔═╡ d9fab336-219e-4fbc-ad48-bbdc3d852836
ex3[:bar2][:, :, 1, 1]

# ╔═╡ 0bc4498d-2335-4aa7-8086-7ab274999b85
chns[1,10:12, 1]

# ╔═╡ b461dbf1-7a58-4f61-97a0-b4cf475d232b
idata = inferencedata(sm)

# ╔═╡ a01c8141-1f9c-40bd-9668-232d65493185
idata.posterior.x

# ╔═╡ 2368d6f7-1d0e-43b6-a565-98fa2059e060
idata.posterior.bar3

# ╔═╡ f1ba91f5-a58b-4fb1-a295-bdb0a305bab0
md" ##### Compare with above df."

# ╔═╡ ead6db29-8176-49ad-9fff-99e2c6d450f4
Array(idata.posterior.bar3[1, 1, 1, :])

# ╔═╡ a47cd679-7228-4135-91fa-d6c3b192b896
df[1, :]

# ╔═╡ 3a0f75b7-6862-4d63-a797-64936caeee49
id3 = [idx for (i, idx) in gm[:bar3]]

# ╔═╡ b16b0a91-77d3-44bd-9977-7c31eb89f8be
begin
	t = (1)
	t2 = (2, 3)
	t3 = (t, t2...)
end

# ╔═╡ 688363b6-f32d-478b-a92b-c673c49ae8fd
display(gm)

# ╔═╡ 6c48113f-ac8d-405e-ba0f-ae1fcdca82c6
gme = gm[:bar3]

# ╔═╡ 50f627f2-a54b-470e-94f7-f0a1982dabfd
id = [idx for (i, idx) in gme]

# ╔═╡ b1bd8e79-b540-431e-a9e7-2c0b18bbf15a
length(id)

# ╔═╡ 04dadc00-8a17-4763-b9a8-840908179625
begin
	str = "("
	no_el = 0
	cdim = 1
	for (i, idx) in enumerate(id)
		println([i, no_el, cdim])
		if cdim == 1 && length(id[i]) == 1
			no_el = no_el + 1
			if no_el == 1
				str = str * "$(gme[i][1])"
			else
				str = str * ",$(gme[i][1])"
			end
		elseif cdim == 1 && length(id[i]) == 2
			no_el += 1
			cdim = 2
			str = str * ",($(gme[i][1])"
		elseif cdim == 1 && length(id[i]) == 3
			no_el += 1
			cdim = 3
			str = str * ",(($(gme[i][1])"
		elseif cdim == 2 && length(id[i]) == 2
			no_el += 1
			str = str * ",$(gme[i][1])"
		elseif cdim == 2 && length(id[i]) == 3
			no_el += 1
			cdim = 3
			str = str * ",($(gme[i][1])"
		elseif cdim == 2 && length(id[i]) == 1
			no_el += 1
			cdim = 1
			str = str * "),$(gme[i][1])"
		elseif cdim == 3 && length(id[i]) == 2
			no_el += 1
			cdim = 2
			str = str * "),$(gme[i][1])"
		elseif cdim == 3 && length(id[i]) == 3
			no_el += 1
			str = str * ",$(gme[i][1])"
		elseif cdim == 3 && length(id[i]) == 1
			no_el += 1
			cdim = 1
			str = str * ")),$(gme[i][1])"
		end
	end
	for i in 1:cdim
		str = str * ")"
	end
	str
end

# ╔═╡ 515ec244-9fa9-470a-87bb-797f3732d533
eval(Meta.parse(str))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
InferenceObjects = "b5cf5a8d-e756-4ee3-b014-01d49d192c00"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
NamedTupleTools = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
StanSample = "c1514b29-d3a0-5178-b312-660c88baa699"

[compat]
DataFrames = "~1.6.1"
InferenceObjects = "~0.3.13"
JSON = "~0.21.4"
NamedTupleTools = "~0.14.3"
StanSample = "~7.5.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0-rc1"
manifest_format = "2.0"
project_hash = "b310df9a9bd6e4831a5331148cfa8a184efbf849"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "16267cf279190ca7c1b30d020758ced95db89cd0"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.5.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompatHelperLocal]]
deps = ["DocStringExtensions", "Pkg", "UUIDs"]
git-tree-sha1 = "be25ab802a22a212ce4da944fe60d7c250ddcfe1"
uuid = "5224ae11-6099-4aaa-941d-3aab004bd678"
version = "0.1.25"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DimensionalData]]
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "Dates", "Extents", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "PrecompileTools", "Random", "RecipesBase", "SparseArrays", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "d61da1be32a0aae538c1412475850dbafe6b0af0"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.25.6"

    [deps.DimensionalData.extensions]
    DimensionalDataMakie = "Makie"

    [deps.DimensionalData.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InferenceObjects]]
deps = ["Compat", "Dates", "DimensionalData", "Requires", "Tables"]
git-tree-sha1 = "6b4b8905afc5556eef491a2e2213bff0207e9bc2"
uuid = "b5cf5a8d-e756-4ee3-b014-01d49d192c00"
version = "0.3.13"

    [deps.InferenceObjects.extensions]
    InferenceObjectsMCMCDiagnosticToolsExt = ["MCMCDiagnosticTools", "Random"]
    InferenceObjectsNCDatasetsExt = "NCDatasets"
    InferenceObjectsPosteriorStatsExt = ["PosteriorStats", "StatsBase"]

    [deps.InferenceObjects.weakdeps]
    MCMCDiagnosticTools = "be115224-59cd-429b-ad48-344e309966f0"
    NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
    PosteriorStats = "7f36be82-ad55-44ba-a5c0-b8b5480d7aa5"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "3f43c2aae6aa4a2503b05587ab74f4f6aeff9fd0"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StanBase]]
deps = ["CSV", "DataFrames", "DelimitedFiles", "Distributed", "DocStringExtensions", "JSON", "NamedTupleTools", "OrderedCollections", "Parameters", "Random", "Unicode"]
git-tree-sha1 = "69c914d59381c8ae48264df426cc8faaf266449e"
uuid = "d0ee94f6-a23d-54aa-bbe9-7f572d6da7f5"
version = "4.8.3"

[[deps.StanSample]]
deps = ["CSV", "CompatHelperLocal", "DataFrames", "DelimitedFiles", "Distributed", "DocStringExtensions", "JSON", "LazyArtifacts", "NamedTupleTools", "OrderedCollections", "Parameters", "Random", "Reexport", "Requires", "Serialization", "StanBase", "TableOperations", "Tables", "Unicode"]
git-tree-sha1 = "2c306b43bec258b6d44cd17f27051a1792f70b47"
uuid = "c1514b29-d3a0-5178-b312-660c88baa699"
version = "7.5.1"

    [deps.StanSample.extensions]
    AxisKeysExt = "AxisKeys"
    InferenceObjectsExt = "InferenceObjects"
    MCMCChainsExt = "MCMCChains"
    MonteCarloMeasurementsExt = "MonteCarloMeasurements"

    [deps.StanSample.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    InferenceObjects = "b5cf5a8d-e756-4ee3-b014-01d49d192c00"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═86e386a0-b56f-42f1-a6de-1f15425d1a59
# ╠═c706075a-0174-450d-a1b0-b202cee4d216
# ╠═4530e47d-abe2-4521-ba6f-1e2e4a46cf3a
# ╠═891015c3-8539-45e1-9a9a-71acfed9cfdf
# ╠═79dca9f3-c2d8-430c-b0d5-658638dcf901
# ╠═bf493f07-d0da-4097-853b-78c9449e4539
# ╠═c11b5f61-be28-48f0-9da8-8b8336222dc4
# ╠═d0e4265b-880a-432c-bfc5-04a2a0b7c8c0
# ╠═037946cf-4622-4b8c-8479-d44451ccbfb1
# ╠═0cd5ee24-b6a5-405e-8c83-d731db58472a
# ╠═f72a0bb0-aa13-4b19-9755-3a30604aa2c0
# ╠═267a7b22-6238-40b1-877f-87e1928d96d8
# ╠═9b4d21c6-b16d-40a3-87d3-7131613a7a28
# ╠═6e4c6e9f-22d5-4ba4-9e8e-43cb46a3d7fe
# ╠═965a04e5-bbd8-4184-b30c-f32b5ddcbeee
# ╠═79643384-8013-41f9-bf32-4f0432ba139a
# ╠═d7be571d-9348-4720-854e-9f4342de6f01
# ╠═6f1a1f67-38c0-4c13-bd2c-dd061e4116ac
# ╠═824db4b1-580e-4dfd-ac1a-c628acda6909
# ╠═df11a159-d82a-4c3a-93ee-a5a5a5e64fc4
# ╠═e6456796-0616-4911-a89f-294684596d70
# ╠═72ab49ea-84af-4371-90b5-9d0c20803db1
# ╠═1af6da5f-2444-47c2-8a2c-d4f79b91e2a7
# ╠═33c784cb-1e1d-4db5-9262-7e8efdb258b8
# ╠═a269c980-b52d-4e60-9fda-bfb6a16620b1
# ╠═744b4729-29ee-4f53-8ba8-c8a8500179ed
# ╠═0802de94-e5a1-46f3-b0a6-259d51e7f4ae
# ╠═87379808-c6cd-4693-aceb-31c9d9ced30d
# ╠═3fadbfa4-2076-4c9b-9e43-4fe0da0d08ce
# ╠═7157a231-7a49-4a99-b597-a378e61c28cf
# ╠═2bf45387-003a-47e4-9669-5e42929b9523
# ╠═a5257168-d4c3-4125-9f57-0dcd1ae38789
# ╠═4d5beb0f-fb84-46ca-ade8-cd83421ee1e6
# ╠═408389bd-7126-499c-ac94-62473643573a
# ╠═0f3658c2-c417-4f45-8721-1b0fd767b1ca
# ╠═da901259-1ac6-44b4-92e3-43b024b95e65
# ╠═055ff913-e461-4027-a5c6-059941de8dcf
# ╠═06502f74-7a3b-4c55-8e51-8211ee574598
# ╠═c7eef545-e660-42de-8e8f-d382e89fd592
# ╠═def76ee3-b8fb-46a0-9f37-262e914cd8d2
# ╠═d9fab336-219e-4fbc-ad48-bbdc3d852836
# ╠═0bc4498d-2335-4aa7-8086-7ab274999b85
# ╠═b461dbf1-7a58-4f61-97a0-b4cf475d232b
# ╠═a01c8141-1f9c-40bd-9668-232d65493185
# ╠═2368d6f7-1d0e-43b6-a565-98fa2059e060
# ╠═f1ba91f5-a58b-4fb1-a295-bdb0a305bab0
# ╠═ead6db29-8176-49ad-9fff-99e2c6d450f4
# ╠═a47cd679-7228-4135-91fa-d6c3b192b896
# ╠═3a0f75b7-6862-4d63-a797-64936caeee49
# ╠═b16b0a91-77d3-44bd-9977-7c31eb89f8be
# ╠═688363b6-f32d-478b-a92b-c673c49ae8fd
# ╠═6c48113f-ac8d-405e-ba0f-ae1fcdca82c6
# ╠═50f627f2-a54b-470e-94f7-f0a1982dabfd
# ╠═b1bd8e79-b540-431e-a9e7-2c0b18bbf15a
# ╠═04dadc00-8a17-4763-b9a8-840908179625
# ╠═515ec244-9fa9-470a-87bb-797f3732d533
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
