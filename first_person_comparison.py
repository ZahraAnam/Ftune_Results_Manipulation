import glob
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


op_indexing_struct = ["person","kitchenware","screen","animal","toy","book","food","object"]

props = dict(boxstyle='round',facecolor='wheat', alpha=0.5)
markers = ["o","^","s","p","P",'*',"h","H","+","x","X"]
color = ['tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan']
ttl_fontsize = 30

ttl_fontweight = "heavy"
sub_ttl_fontsize = 28
sub_ttl_color = "purple"
sub_ttl_fontweight = "demi"
label_fnt = 26
scores = np.arange(0.5,0.01)

first_person_counter = 0
secnd_person_counter = 0
person_det = 0

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def comparator(coco_in,yolo_in):

	global first_person_counter
	global secnd_person_counter
	global person_det

	#print("In Comparator")
	
	coco_inp = coco_in.reshape(-1,3)
	yolo_inp = yolo_in.reshape(-1,3)

	for coco_data in coco_inp:
		coco_bb = coco_data[1]
		p_id = coco_data[2]

		
		for yolo_data in yolo_inp:
			
			yolo_bb =yolo_data[1]

			iou = bb_intersection_over_union(coco_bb,yolo_bb)
			if iou >= 0.2 and p_id==1:
				first_person_counter+=1
				
			elif iou >= 0.2 and p_id!=1:
				secnd_person_counter+=1
			if iou>=0.2:
				person_det+=1

		
	return



if __name__ == "__main__":

	

	#coco_fname = "comb_data.json"
	yolo_fname = "best_predictions.json"

	#coco_fl = os.path.join(inp_dir,coco_fname)
	#coco_fl = "/media/anam_zahra/QuantEx/QuantEx_Data/Sample_Dataset/PID_test_data.json"

	coco_fl = "/home/anam_zahra/Codes/Evaluation_Codes/Ftune_Results_Manipulation/comb_data.json"

	#yolo_fl = os.path.join(inp_dir,yolo_fname)

	yolo_fl = "/home/anam_zahra/Codes/yolov5/runs/val/ft_50_full_test/best_predictions.json"
	#yolo_fl = "/home/anam_zahra/Codes/yolov5/runs/detect/exp/coco_labels/yolo.json"
	#yolo_fl = "/home/anam_zahra/Codes/yolov5/runs/detect/yolov5l/coco_labels/yolo.json"

	with open(coco_fl) as json_data:
		coco_data = json.load(json_data)

	with open(yolo_fl) as json_data:
		yolo_data = json.load(json_data)

	coco_df = pd.DataFrame(coco_data)
	yolo_df = pd.DataFrame(yolo_data)

	# Data Pruning - Removing NAN rows and removing all classes except people

	coco_df = coco_df.drop(coco_df[coco_df.image_id == "NAN"].index)
	coco_df = coco_df.drop(["area","iscrowd","id"],axis=1)

	coco_df = coco_df.drop(coco_df[coco_df.category_id != 1].index)
	coco_df["category_id"] = coco_df["category_id"].astype(int)

	coco_df.set_index(["image_id"],inplace=True)

	yolo_df = yolo_df.drop(yolo_df[yolo_df.category_id!=0].index)

	yolo_df = yolo_df.drop(yolo_df[yolo_df.score<0.01].index)

	yolo_df["category_id"] = yolo_df["category_id"]+1

	yolo_df.set_index(["image_id"],inplace=True)
	yolo_df_ind = yolo_df.index.tolist()


	#coco_ind = set(coco_df.index.to_list())



	for idx,row in coco_df.groupby(level=0):
		im_id = row.index.unique().item()
		#print("Resolving image_id = ", im_id,"\n")
		coco_inp = coco_df.loc[im_id]
		coco_arr = coco_inp.to_numpy()

		if im_id in yolo_df_ind:
			yolo_inp = yolo_df.loc[im_id]
			yolo_arr = yolo_inp.to_numpy()
			comparator(coco_arr,yolo_arr)
		
		
		
	print("\n"*3)
	
	print("*" * 10)

	print("People Count in GT ",coco_df["category_id"].value_counts(),"\n")
	print("People Detected ", yolo_df["category_id"].value_counts(),"\n")
	print("People bboxes with iou>=0.2 ",person_det,"\n")
	
	print("P_ID Count",coco_df["Person_ID"].value_counts())

	#print(sum(coco_df["Person_ID"].value_counts()))
	
	print("First Person Detection Count = " ,first_person_counter,"\n")
	print("Second Person Detectiob Count = ", secnd_person_counter,"\n")

	P_id_count = dict(coco_df["Person_ID"].value_counts())
	gt_secnd_prsn_cnt = 0
	for k,v in P_id_count.items():
		if k!=1.0:
			gt_secnd_prsn_cnt+=v
	gt_dict = {}
	dt_dict = {}
	gt_dict.update({"total_people":int(coco_df["category_id"].value_counts().values),"first_person":P_id_count[1.0],"second_person":gt_secnd_prsn_cnt})
	dt_dict.update({"total_people":person_det,"first_person":first_person_counter,"second_person":secnd_person_counter})
	print("*" * 10)
	inp_keys = list(gt_dict.keys())
	tk = np.asarray([i for i in range(len(inp_keys))])
	wd = 0
	fig,ax = plt.subplots(1,1,figsize=(20,20))
	fig.suptitle("Overview of People Occurence and Yolo Detection",fontsize=ttl_fontsize,fontweight = ttl_fontweight)
	fig.text(0.75,0.92,"Confidence >=0.01, IoU>=0.2", fontsize=14,verticalalignment='top',bbox=props)
	fig.supylabel("Frequency",fontsize = label_fnt,fontweight=sub_ttl_fontweight)
	fig.supxlabel("People Dist",fontsize = label_fnt,fontweight=sub_ttl_fontweight)
	for cnt,k in enumerate(inp_keys):
		ax.bar(cnt,gt_dict[k],width=0.20,color=color[0])
		ax.bar(cnt+0.20,dt_dict[k],width=0.20,color=color[1])
		wd+=0.20*2
		plt.pause(0.05)

	plt.xticks(tk,gt_dict.keys())
	plt.legend(["Ground Truth","Detections"])
	plt.show()