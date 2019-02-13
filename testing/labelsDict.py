import pickle

subjects = [1,5,6,7,8]
actions = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
subactions = [1,2]    
mov_dict=dict()
mov_dict["label_names"]=list()

for subject in subjects:
	for action in actions:
		for subaction in subactions:
			folder_name_no_cam = 'S{:02d}/Act{:02d}/Subact{:02d}/'.format(subject, action, subaction)
			mov_dict["label_names"].append(folder_name_no_cam)

pickle_out = open("mov_dict.pickle","wb")
pickle.dump(mov_dict, pickle_out)
pickle_out.close()

for key,value in mov_dict.items():
	print(key,value)




