using FileIO
using Images
using Flux
using Functors

import LinearAlgebra
import Statistics
import Random

global _dataset_path = "./mango_images"

struct MangoData
    cls_file_url::Vector{Vector{String}}    # image file url for each class 
    cls_name::Vector{String}                # classname 
    cls_datacount::Vector{Integer}          # datacount for each class
    n::Integer                              # total image count
end 

function buildMangoDataFold(fold_no::Int; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)
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
    testdataFold = MangoData(imgpath_cls_vec_test, cls_name, cls_datacount_test, sum(cls_datacount_test))

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
    trainingDataFold = MangoData(imgpath_cls_vec_train, cls_name, cls_datacount_train, sum(cls_datacount_train))

    return trainingDataFold, testdataFold
end

# datastore by fold number
struct dataFold
    training_img::MangoData
    testing_img::MangoData
end

## Init Dataset ##
# for rebuild or re-locate dataset directory
function initMangoDataset(;dataset_path=_dataset_path)
    global _classname = readdir(_dataset_path)       # class name given by subfolders
    global _classcount = length(_classname)
    
    # load all images path by class
    class_imgpath = Vector{Vector{String}}(undef, _classcount)
    for i = 1:_classcount
        class_imgpath[i] = readdir(joinpath(_dataset_path, _classname[i]), join=true)
    end
    
    global _dataset_imgcount = sum(length.(class_imgpath))
    
    println("Dataset dir: $_dataset_path")
    println("Dataset Class Count: $_classcount")
    println("Dataset Image Count: $_dataset_imgcount")

    # create datafold
    global dataFold_1 = dataFold(buildMangoDataFold(1; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)...)
    global dataFold_2 = dataFold(buildMangoDataFold(2; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)...)
    global dataFold_3 = dataFold(buildMangoDataFold(3; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)...)
    global dataFold_4 = dataFold(buildMangoDataFold(4; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)...)
    global dataFold_5 = dataFold(buildMangoDataFold(5; kfold=5, n_cls=5, cls_name=_classname, class_imgpath=class_imgpath)...)
end

# Initialize This Dataset
initMangoDataset();
##################

# Function: dispatch image by type(training or testing)
# mutable struct for recording the data index 
# Flux one-hot encoding for class label
mutable struct MangoDispatcher
    mangodata::MangoData
    
    # class number selection
    cls_current_index::Integer
    cls_index_vector::Vector{Int}               
    cls_max_index::Integer
    
    # in-class file url selection 
    incls_current_index::Vector{Int}             
    incls_index_vector::Vector{Vector{Int}}
    incls_max_index::Vector{Int}   

    shuffle_enable::Bool
    minibatch_size::Integer
    no_class::Integer
    outputsize::Vector{Int}
end

@functor MangoDispatcher
# constructor / modifier
function MangoDispatcher(mangofold::MangoData, output_size=[224,224], dispatch_size=1,
    shuffle_enable=true, classcount=_classcount)

    cls_index_vector = Int[]    # class-pick indexing vector
    incls_index_vector = Vector{Vector{Int}}(undef, classcount)   # in-class pick indexing vector
    for cls_no = 1:classcount
        cls_index_vector = cat(cls_index_vector, repeat([cls_no,], outer=mangofold.cls_datacount[cls_no]), dims=1)
        incls_index_vector[cls_no] = 1:mangofold.cls_datacount[cls_no]
    end

    if shuffle_enable
        Random.shuffle!(cls_index_vector)
        Random.shuffle!.(incls_index_vector)
    end

    return MangoDispatcher(
        mangofold,
        0, cls_index_vector, mangofold.n,
        zeros(classcount), incls_index_vector, mangofold.cls_datacount,
        shuffle_enable, dispatch_size, classcount, output_size
    );
end

# reset current_index and re-shuffle
function resetMangoDispatcher!(x::MangoDispatcher;shuffle_enable=x.shuffle_enable)
    x.current_index = 0;
    if shuffle_enable
        Random.shuffle!(x.cls_index_vector)
        Random.shuffle!.(x.incls_index_vector)
    end
    return x
end

function (x::MangoDispatcher)()
    x.cls_current_index += 1 # update index pointer
    if (x.cls_current_index + x.minibatch_size) <= x.cls_max_index
        
        select_cls = x.cls_index_vector[x.cls_current_index]    # select class
        x.incls_current_index[select_cls] += 1                  # update index pointer
        
        # select, load  image file
        img_filename = x.mangodata.cls_file_url[select_cls][x.incls_current_index[select_cls]]
        temp_img = Images.load(img_filename);
        temp_img_pp = Images.imresize(temp_img, x.outputsize...) |> Images.channelview; 
        # Images.channelview() = Channel x W x H 
        loaded_img = PermutedDimsArray(copy(temp_img_pp), (2, 3, 1));   # W.H.C
        loaded_img = Flux.unsqueeze(loaded_img, 4);                     # W.H.C.N

        # load label 
        loaded_label = Flux.onehot()

        if x.minibatch_size > 1
            for i = 2:x.minibatch_size
                x.cls_index_vector += 1
                select_cls = x.cls_index_vector[x.cls_current_index]    # select class
                x.incls_current_index[select_cls] += 1                  # update index pointer

                temp_img = Images.load(data_url[select_indx]);
                temp_img_pp = Images.imresize(temp_img, w, h);
                temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
                temp_img_pp = PermutedDimsArray(temp_img_pp, (2, 3, 1)); # W.H.C
                temp_img_pp = Flux.unsqueeze(temp_img_pp, 4); # W.H.C.N
                loaded_img = cat(temp_img_pp, loaded_img; dims=4); # concatenate loaded image along dims=4 dimension
                loaded_img = copy(loaded_img);
            end
        end

    end

    outputDL = Flux.Data.DataLoader(
        (loaded_img, cls_label),
        batchsize=x.minibatch_size,
        shuffle=x.shuffle_enable    
    );

    return outputDL
end