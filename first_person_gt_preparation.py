import glob
import os
import sys
import json
import numpy as np
from numpyencoder import NumpyEncoder
import pandas as pd
import warnings


test_im_counter = 0
test_id_counter = 0
test_data_dict = {}
test_data_dict["annotations"] = []
test_loc_counter = 0

op_indexing_struct = ["person","kitchenware","screen","animal","toy","book","food","object"]

inp_dir = "/home/anam_zahra/Codes/Python_Codes/Ftune_Comparison_Home_Work/Raw_Annotations/AgeGroup_004/257511/Home/"



def op_index_structuring(df_in):
    """
    Given a dataframe with column category_id that contains the name of class
    Returns the category_id assigned as per the op_indexing_struct 
    First category_id is set to 1.
    """
    for nr,val in enumerate(op_indexing_struct):
        df_in["category_id"] = df_in["category_id"].replace(val,(nr+1))
    return df_in

def categorization(data):
    df_categ = pd.DataFrame(data['categories'])

    

    df_annot = pd.DataFrame(data['annotations'])
    # print(df_annot.head(3))
    
    for i in df_categ.index:
        c_id = df_categ.loc[i, "id"]
        c_name = df_categ.loc[i, "name"]
        
        df_annot['category_id'] = df_annot['category_id'].replace(c_id,c_name)

    
    #df_annot["image_id"] = (df_annot["image_id"])-1
    df_annot = df_annot.drop(df_annot[df_annot.category_id == "noise"].index)
    df_annot['category_id'] = df_annot['category_id'].replace('cuttlery','kitchenware')
    df_annot['category_id'] = df_annot['category_id'].replace('Reflection','person')

    
    
    df_annot = op_index_structuring(df_annot)

    
    return df_annot

def del_dict_keys(inp_dict,key_to_keep):
    """
    Deletes all the keys except the one to keep
    """
    for k, v in list(inp_dict.items()):
        if k != key_to_keep:
            del(inp_dict[k])
            continue   
        #print(k)


    return inp_dict


def test_data_prep(inp_list):
    # Calling global variables to uodate
    global test_im_counter
    global test_id_counter
    global test_data_dict
    global test_loc_counter

    # Looping through every folder to unzip it and process
    for fld in inp_list:
        """
        print("Writing for the " + fld, "\n"*2)
        op_loc = os.path.join(zip_data_dir,fld)
        op_fld = os.path.join(base_dir,"temp")
        if not os.path.exists(op_fld):
            os.makedirs(op_fld)
        
        #folder_unzipper(op_loc,op_fld)
        """
        #################################
        annot_fl = os.path.join(inp_dir,fld,"instances_default.json")
        with open(annot_fl) as json_data:
            data = json.load(json_data)
        
        # Removing attributes for current process
        """
        for ann in data["annotations"]:
            #ann = ann["attributes"].pop(["",])
            ann["attributes"] = del_dict_keys(ann["attributes"],"ID")
            
        """   

        #df = pd.DataFrame(data["annotations"])

        #df = df.drop(["attributes","segmentation","area"],axis=1)
        # Calling a function to remove noise and merge certain classes as well as assign same category_ids to every annotation file
        df_annot = categorization(data)
        df_annot["Person_ID"] = "NAN"
        #df_annot["ID"] = df_annot["ID"].astype(int)
        
        

        for cnt,row in df_annot.iterrows():
            #ann = ann["attributes"].pop(["",])
            
            ann = row["attributes"]
            if "ID" in ann.keys():
                p_id = ann["ID"]
                """
                att = del_dict_keys(ann,"ID")
                for k,v in att.items():
                    p_id = int(v)
                """
                df_annot.loc[cnt,"Person_ID"] = p_id

        df_annot = df_annot.drop(["segmentation","attributes"],axis=1) 

        #df_annot = df_annot.drop(df_annot[df_annot.image_id == "NAN"].index)

        # Getting unique image ids to write image element only once in .json file
        imz = list(df_annot["image_id"].unique())
        # Getting max values in order to update the naming sequence and count as all images are gonna be put in one folder
        im_updater = max(imz)
        id_updater = max(df_annot["id"].unique())

        for idx,row in df_annot.iterrows():
            # Getting new id as per update
            im_id = row["image_id"] - 1  + test_im_counter
            #id = row["id"] + test_id_counter

            """
            # Renaming and movingf image and updating the images section of the dict
            op_fname = str(im_id).zfill(6) + ".PNG"
            in_fname = data["images"][row["image_id"]-1]["file_name"]
            in_fl = os.path.join(op_fld,"images",in_fname)
            op_fl = os.path.join(out_test_im_folder,op_fname)
            # Processing image file only once 
            if os.path.isfile(in_fl):
                os.rename(in_fl,op_fl)
                im_width = data["images"][row["image_id"]-1]["width"]
                im_height = data["images"][row["image_id"]-1]["height"]
                # id for images section is same as im_id
                test_data_dict["images"].append({
                                            "id" : im_id,
                                            "width" : im_width,
                                            "height" : im_height,
                                            "file_name" : op_fname

                                    })
                test_loc_counter+=1
            """

            # Annotations - Updating annotation dict
            
            test_data_dict["annotations"].append({
                                            "id" : row["id"] + test_id_counter,
                                            "image_id" : im_id,
                                            "category_id" : row["category_id"],
                                            "area" : row["area"],
                                            "bbox" : row["bbox"],
                                            "iscrowd" : row["iscrowd"],
                                            "Person_ID" : row["Person_ID"]

            })


        # Updating counters 
        test_im_counter  += im_updater
        test_id_counter += id_updater

        #shutil.rmtree(op_fld)

    return


if __name__ == "__main__":

    """

    annot_file = "/media/anam_zahra/QuantEx/QuantEx_Data/Processed_Data/Annotations/AgeGroup_004/257511/coco_gt.json"
    annot_file = "/media/anam_zahra/QuantEx/QuantEx_Data/Processed_Data/Annotations/AgeGroup_004/257511/Home/001/ground_truth_8180.json"
    with open(annot_file) as json_data:
        annot_data = json.load(json_data)

        print(annot_data.keys())
    #images_list = annot_data["images"]
    cats_list = annot_data["categories"]
    annots_list = annot_data["annotations"]
    cat_df = pd.DataFrame(cats_list)
    annots_df = pd.DataFrame(annots_list)
    print(cat_df)
    print("\n" * 2)
    print(annots_df.head(3))


    pred_dir = "/media/anam_zahra/QuantEx/QuantEx_Data/Processed_Data/Ftune_Predictions/Yolo"
    pred_files = os.listdir(pred_dir)

    """

    #inp_dir = "/home/anam_zahra/Codes/Python_Codes/Ftune_Comparison_Home_Work/Raw_Annotations/AgeGroup_004/257511/Home/"

    inp_dir = "/media/anam_zahra/PortableSSD/257511/Home"


    #out_dir = "/home/anam_zahra/Codes/Python_Codes/Ftune_Comparison_Home_Work/Raw_Annotations/AgeGroup_004/257511/"

    out_dir = "/home/anam_zahra/Codes/Evaluation_Codes/Ftune_Results_Manipulation/"

    inp_folders = os.listdir(inp_dir)

    test_data_prep(inp_folders)

    
    out_file = os.path.join(out_dir,"comb_data.json")

    with open(out_file,'w') as json_data:
        json.dump(test_data_dict,json_data,cls=NumpyEncoder)
   

