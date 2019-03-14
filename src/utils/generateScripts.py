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
"""
layerDim=["2"]
hiddenDim=["512"]
#hiddenDim=["64","128","256","512","1024","2048"]
#hiddenDim=["64","256", "1024"]
#hiddenDim=["128","512","2048"]

for i in layerDim:
    for j in hiddenDim:
        exp = "experiments/lstm/lstmConfig/layerDim" + i + "/hiddenDim" + j + "/"
        print("clear;rm -r " + exp + "events/ " + ";rm -r " + exp + "gifs/;. experiments/prepare_session.sh 0,1,2;python src/train.py --exp_dir " + exp + ";")
"""


#subSampling=["window300","odd"]
subSampling=["window200","even"]

for i in subSampling:
    exp = "experiments/lstm/subSampling/" + i + "/"
    print("clear;rm -r " + exp + "events/ " + ";rm -r " + exp + "gifs/;. experiments/prepare_session.sh 0,1,2,3;python src/train.py --exp_dir " + exp + ";")


"""
#seqDim=["49","149"]
seqDim=["199","99"]

for i in seqDim:
    exp = "experiments/lstm/seqDim/d" + i + "/"
    print("clear;rm -r " + exp + "events/ " + ";rm -r " + exp + "gifs/;. experiments/prepare_session.sh 0,1,2,3;python src/train.py --exp_dir " + exp + ";")
"""