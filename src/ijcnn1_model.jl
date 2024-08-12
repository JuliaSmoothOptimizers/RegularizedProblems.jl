export ijcnn1_generate_data_tr, ijcnn1_generate_data_test, ijcnn1_model_sto

#packages required for getting and decompressing IJCNN1 from LIBSVM
using HTTP, CodecBzip2

"""
    ijcnn1_generate_data_tr(; n::Int = 49990, d::Int = 22)

Generates data of IJCNN1 training set from LIBSVM : a data matrix and a vector of labels in {-1, 1}ⁿ

### Keyword arguments

* n::Int : Number of features
* m::Int : Number of data

For IJCNN1 training, m = 49990 and n = 22.

### Return values

* A::Matrix{Float64} : IJCNN1 train data matrix
* y::Vector{Float64} : IJCNN1 train labels
"""
function ijcnn1_generate_data_tr(; n::Int = 49990, d::Int = 22)
    #getting data
    data = HTTP.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2")
    bz2_file = transcode(Bzip2Decompressor, data.body)
    data_file = String(bz2_file)
    lines = split(data_file, '\n')

    #allocate memory to store data
    A = zeros(Float64, n, d)
    y = zeros(Float64, n)
    i = 1
    while i ≤ n
        dummy = split(lines[i], ' ')
        y[i] = parse(Float64, dummy[1])
        for j in 2:(length(dummy))
            loc_val = split(dummy[j],':')
            if length(loc_val) < 2
                display(i)
                display(loc_val)
            end
            A[i, parse(Int, loc_val[1])] = parse(Float64, loc_val[2])
        end
    i += 1
    end
    A, y
end

"""
    ijcnn1_generate_data_test(; n::Int = 91701, d::Int = 22)

Generates data of IJCNN1 testing set from LIBSVM : a data matrix and a vector of labels in {-1, 1}ⁿ

### Keyword arguments

* n::Int : Number of features
* m::Int : Number of data

For IJCNN1 test, m = 91701 and n = 22.

### Return values

* A::Matrix{Float64} : IJCNN1 train data matrix
* y::Vector{Float64} : IJCNN1 train labels
"""
function ijcnn1_generate_data_test(; n::Int = 91701, d::Int = 22)
    #getting data
    data = HTTP.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2")
    bz2_file = transcode(Bzip2Decompressor, data.body)
    data_file = String(bz2_file)
    lines = split(data_file, '\n')

    #allocate memory to store data
    A = zeros(Float64, n, d)
    y = zeros(Float64, n)
    i = 1
    while i ≤ n
        dummy = split(lines[i], ' ')
        y[i] = parse(Float64, dummy[1])
        for j in 2:(length(dummy))
            loc_val = split(dummy[j],':')
            if length(loc_val) < 2
                display(i)
                display(loc_val)
            end
            A[i, parse(Int, loc_val[1])] = parse(Float64, loc_val[2])
        end
    i += 1
    end
    A, y
end

"""
    ijcnn1_train_model()

Model constructor of IJCNN1 training problem.

### Return values

* nlp::FirstOrderNLPModel : an instance of IJCNN1 train with NLP structure
* nls::FirstOrderNLSModel : an instance of IJCNN1 train with NLS structure
* sol::Vector{Float64} : IJCNN1 train solution i.e. its given labels
"""
function ijcnn1_train_model()
    A_tr, b_tr = ijcnn1_generate_data_tr()
    svm_model(A_tr', b_tr)
end


"""
    ijcnn1_train_model()

Model constructor of IJCNN1 testing problem.

### Return values

* nlp::FirstOrderNLPModel : an instance of IJCNN1 test with NLP structure
* nls::FirstOrderNLSModel : an instance of IJCNN1 test with NLS structure
* sol::Vector{Float64} : IJCNN1 test solution i.e. its given labels
"""
function ijcnn1_test_model()
    A_test, b_test = ijcnn1_generate_data_test()
    svm_model(A_test', b_test)
end