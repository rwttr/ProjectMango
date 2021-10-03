using FileIO
using Images
using Flux
using Functors

import LinearAlgebra
import Statistics
import Random

dataset_path = "./mango_images"

classname = readdir(dataset_path)       # class name given by subfolders
no_class = length(classname)

# load all images path by class
class_imgpath = Vector{Vector{String}}(undef, no_class)
for i = 1:no_class
    class_imgpath[i] = readdir(joinpath(dataset_path, classname[i]), join=true)
end

no_img = sum(length.(class_imgpath))

println("Dataset dir: $dataset_path")
println("Dataset Class Count: $no_class")
println("Dataset Image Count: $no_img")

# for this project = 5 class
struct MangoData
    cls_l::Vector{String}           # vector of image path (string) 
    cls_low::Vector{String}
    cls_m::Vector{String}
    cls_s::Vector{String}
    cls_reject::Vector{String}
    cls_datacount::Vector{Integer}
    n::Integer
end

function buildMangoDataFold(fold_no::Int; kfold=5, n_cls=5, class_imgpath=class_imgpath)
    # largest cls size
    cls_len = length.(class_imgpath) 
    kfold_comlen = maximum(cls_len)
    kfold_label = repeat(1:kfold;outer=Int(ceil(kfold_comlen / kfold)))
    
    # testing data pick
    testdata_indexer = findall(x -> x == fold_no, kfold_label)
    imgpath_cls_vec_test = Vector{Union{Nothing,Vector{String}}}(nothing, n_cls)
    cls_datacount_test = zeros(Integer, n_cls)
    for cls_no in 1:n_cls
        imgpath_cls_temp = String[];
        for i in testdata_indexer
            if (i <= cls_len[cls_no])
                push!(imgpath_cls_temp, class_imgpath[cls_no][i])
            end
        end
        cls_datacount_test[cls_no] = length(imgpath_cls_temp)
        imgpath_cls_vec_test[cls_no] = imgpath_cls_temp
    end
    # testdata fold
    testdataFold = MangoData(imgpath_cls_vec_test..., cls_datacount_test, sum(cls_datacount_test))

    # training data pick
    trainingdata_indexer = findall(x -> x != fold_no, kfold_label)
    imgpath_cls_vec_train = Vector{Union{Nothing,Vector{String}}}(nothing, n_cls)
    cls_datacount_train = zeros(Integer, n_cls)
    for cls_no in 1:n_cls
        imgpath_cls_temp = String[];
        for i in trainingdata_indexer
            if (i <= cls_len[cls_no])
                push!(imgpath_cls_temp, class_imgpath[cls_no][i])
            end
        end
        cls_datacount_train[cls_no] = length(imgpath_cls_temp)
        imgpath_cls_vec_train[cls_no] = imgpath_cls_temp
    end
    # training data fold
    trainingDataFold = MangoData(imgpath_cls_vec_train..., cls_datacount_train, sum(cls_datacount_train))

    return trainingDataFold, testdataFold
end

# datastore by fold number
struct dataFold
    training_img::MangoData
    testing_img::MangoData
end

dataFold_1 = dataFold(buildMangoDataFold(1; kfold=5, n_cls=5, class_imgpath=class_imgpath)...)
dataFold_2 = dataFold(buildMangoDataFold(2; kfold=5, n_cls=5, class_imgpath=class_imgpath)...)
dataFold_3 = dataFold(buildMangoDataFold(3; kfold=5, n_cls=5, class_imgpath=class_imgpath)...)
dataFold_4 = dataFold(buildMangoDataFold(4; kfold=5, n_cls=5, class_imgpath=class_imgpath)...)
dataFold_5 = dataFold(buildMangoDataFold(5; kfold=5, n_cls=5, class_imgpath=class_imgpath)...)

# dispatch image by type(training or testing)
# one-hot encoding for class dispatch
mutable struct MangoDispatcher
    current_MangoFold::MangoData
    current_index::Integer
    index_vector::Vector{Int}
    max_index::Integer
    shuffle_enable::Bool
    minibatch_size::Integer
    no_class::Integer
    outputsize::Vector{Int}
end
@functor MangoDispatcher
# constructor / modifier
function MangoDispatcher(mangofold::MangoData, output_size=[224,224], dispatch_size=1, shuffle_enable=true)
    # calculate indexing vector
    index_vector = Int[]
    for cls_no = 1:5
        index_vector = cat(index_vector, repeat([cls_no,], outer=mangofold.cls_datacount[cls_no]), dims=1)
    end

    if shuffle_enable
        Random.shuffle!(index_vector)
    end
   
    return MangoDispatcher(
        mangofold,
        0,
        index_vector,
        mangofold.datacount,
        shuffle_enable,
        dispatch_size,
        5,
        output_size
    );
end

# reset current_index and re-shuffle
function resetMangoDispatcher!(x::MangoDispatcher;shuffle_enable=x.shuffle_enable)
    x.current_index = 0;
    if shuffle_enable
        Random.shuffle!(x.index_vector)
    end
    return x
end

function dispatchMango(datafold::DataFold; output_size=[224,224], dispatch_size=1, shuffle_enable=true)


    # dispatch image data : loaded_img  
    temp_img = Images.load(data_url[select_indx]);
    temp_img_pp = Images.imresize(temp_img, w, h);
    temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
    loaded_img = copy(temp_img_pp);  
    loaded_img = PermutedDimsArray(loaded_img, (2, 3, 1)); # W.H.C
    loaded_img = Flux.unsqueeze(loaded_img, 4); # W.H.C.N


end    