dropout=["1","3","5"]
#weightDecay=["0","1EMinus4","1EMinus5","5EMinus5"]
weightDecay=["1EMinus5"]
optimizer=["Adam"]
subSampling=["Even"]
input=["HiddenFrames"]

for i in input:
    for j in subSampling:
        for k in optimizer:
            for l in weightDecay:
                for m in dropout:
                    exp="experiments/lstm/input"+i+"/sub"+j+"/optim"+k+"/weightDecay"+l+"/dropout"+m+"/"
                    print("clear;rm -r "+exp+"events/ "+";rm -r "+exp+"gifs/;. experiments/prepare_session.sh 0,1,2;python src/train.py --exp_dir "+exp+";")

