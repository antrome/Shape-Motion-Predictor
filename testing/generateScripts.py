"""
dropout=["5"]
#weightDecay=["0","1EMinus4","1EMinus5","5EMinus5"]
weightDecay=["5EMinus5"]
optimizer=["Adam"]
subSampling=["Window300"]
input=["HiddenFrames"]

for i in input:
    for j in subSampling:
        for k in optimizer:
            for l in weightDecay:
                for m in dropout:
                    exp="experiments/lstm/input"+i+"/sub"+j+"/optim"+k+"/weightDecay"+l+"/dropout"+m+"/"
                    print("clear;rm -r "+exp+"events/ "+";rm -r "+exp+"gifs/;. experiments/prepare_session.sh 0,1,2;python src/train.py --exp_dir "+exp+";")
"""

#lr=["1EMinus3","1EMinus4","1EMinus5","1EMinus6"]
#lr=["5EMinus3","5EMinus4","5EMinus5","5EMinus6"]
#lr=["1EMinus2"]
layerDim=["4"]
#hiddenDim=["64","128","256","512","1024","2048"]
#hiddenDim=["64","256","1024"]
hiddenDim=["128","512","2048"]

for i in layerDim:
    for j in hiddenDim:
        exp = "experiments/lstm/lstmConfig/layerDim" + i + "/hiddenDim" + j + "/"
        print("clear;rm -r " + exp + "events/ " + ";rm -r " + exp + "gifs/;. experiments/prepare_session.sh 0,1,2;python src/train.py --exp_dir " + exp + ";")
