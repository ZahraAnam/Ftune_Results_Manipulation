import glob
import os
import sys
import json
import numpy as np
from numpyencoder import NumpyEncoder
import pandas as pd
import warnings




op_indexing_struct = ["person","kitchenware","screen","animal","toy","book","food","object"]
models = ["yolo","faster_rcnn_X_101_32x8d_FPN_3x"]



def categorize_data(df_in):

	for nr,val in enumerate(op_indexing_struct):
		df_in["category"]=df_in["category"].replace(val,(nr+1))

	return df_in



if __name__ == "__main__":


    inp_dir = "/media/anam_zahra/QuantEx/QuantEx_Data/Processed_Data/Predictions/AgeGroup_004/257511/Home/001/json_outputs"
    op_dir = "/home/anam_zahra/Codes/Ftune_Results_Manipulation/json_files/"

    for md in models:
    	inp_fl = os.path.join(inp_dir,md+".json")

    	with open(inp_fl) as json_data:
    		data = json.load(json_data)

    	pred_df = pd.DataFrame(data["annotations"])
    	
    	pred_df = pred_df.drop("category_id",axis=1)

    	pred_df = categorize_data(pred_df)
    	pred_df.rename(columns = {'category': 'category_id'}, inplace = True)

    	out_dict = {}
    	out_dict["predictions"] = []
    	for idx,row in pred_df.iterrows():
    		out_dict["predictions"].append({	
                                            "image_id" : row["image_id"],
                                            "category_id" : row["category_id"],
                                            "bbox" : row["bbox"],
                                            "score" : row["score"]
                                            

    			})
    	op_fl = os.path.join(op_dir,md+".json")

    	with open(op_fl,'w') as json_data:
    		json.dump(out_dict,json_data,cls=NumpyEncoder)


