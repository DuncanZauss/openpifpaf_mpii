from scipy.io import loadmat, savemat
from PIL import Image
import os
import os.path as osp
import numpy as np
import json

# run this code in the 'mpii_human_pose_v1_u12_2' folder

def check_empty(list,name):
    
    try:
        list[name]
    except ValueError:
        return True

    if len(list[name]) > 0:
        return False
    else:
        return True



def main():
    annot_file = loadmat('MPII/mpii_human_pose_v1_u12_1')['RELEASE']
    reorder_indices = [8, 9, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
    
    for db_type in ["train", "test"]: 
        joint_num = 16
        img_num = len(annot_file['annolist'][0][0][0])
        
        aid = 0
        coco = {'images': [], 'categories': [], 'annotations': []}
        for img_id in range(img_num):
            
            if ((db_type == 'train' and annot_file['img_train'][0][0][0][img_id] == 1) or (db_type == 'test' and annot_file['img_train'][0][0][0][img_id] == 0)) and \
                check_empty(annot_file['annolist'][0][0][0][img_id],'annorect') == False: #any person is annotated
            
                filename = str(annot_file['annolist'][0][0][0][img_id]['image'][0][0][0][0]) #filename
                img = Image.open(osp.join('./MPII/images', filename))
                w,h = img.size
                img_dict = {'id': img_id, 'file_name': filename, 'width': w, 'height': h}
                coco['images'].append(img_dict)
        
                if db_type == 'test':
                    continue
                
                person_num = len(annot_file['annolist'][0][0][0][img_id]['annorect'][0]) #person_num
                # joint_annotated = np.zeros((person_num,joint_num))
                for pid in range(person_num):
                    
                    if check_empty(annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid],'annopoints') == False: #kps is annotated
                        
                        bbox = np.zeros((4)) # xmin, ymin, w, h
                        kps = np.zeros((joint_num,3)) # xcoord, ycoord, vis
        
                        #kps
                        annot_joint_num = len(annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0])
                        for jid in range(annot_joint_num):
                            annot_jid = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['id'][0][0]
                            kps[annot_jid][0] = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['x'][0][0]
                            kps[annot_jid][1] = annot_file['annolist'][0][0][0][img_id]['annorect'][0][pid]['annopoints']['point'][0][0][0][jid]['y'][0][0]
                            kps[annot_jid][2] = 1
                       
                        #bbox extract from annotated kps
                        kps = kps[reorder_indices, :]
                        annot_kps = kps[kps[:,2]==1,:].reshape(-1,3)
                        xmin = np.min(annot_kps[:,0])
                        ymin = np.min(annot_kps[:,1])
                        xmax = np.max(annot_kps[:,0])
                        ymax = np.max(annot_kps[:,1])
                        width = xmax - xmin - 1
                        height = ymax - ymin - 1
                        
                        # corrupted bounding box
                        if width <= 0 or height <= 0:
                            continue
                        # 20% extend    
                        else:
                            bbox[0] = (xmin + xmax)/2. - width/2*1.2
                            bbox[1] = (ymin + ymax)/2. - height/2*1.2
                            bbox[2] = width*1.2
                            bbox[3] = height*1.2
        
        
                        person_dict = {'id': aid, 'image_id': img_id, 'category_id': 1, 'area': bbox[2]*bbox[3], 'bbox': bbox.tolist(), 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(), 'num_keypoints': int(np.sum(kps[:,2]==1))}
                        coco['annotations'].append(person_dict)
                        aid += 1

        category = {
            "supercategory": "person",
            "id": 1,  # to be same as COCO, not using 0
            "name": "person",
            "skeleton": [[0,1],
                [1,2], 
                [2,6], 
                [7,12], 
                [12,11], 
                [11,10], 
                [5,4], 
                [4,3], 
                [3,6], 
                [7,13], 
                [13,14], 
                [14,15], 
                [6,7], 
                [7,8], 
                [8,9]] ,
            
            
          "keypoints": [
              'head_bottom',
              'head_top',
              'left_shoulder',
              'right_shoulder',
              'left_elbow',
              'right_elbow',
              'left_wrist',
              'right_wrist',
              'left_hip',
              'right_hip',
              'left_knee',
              'right_knee',
              'left_ankle',
              'right_ankle',
          ]}    # 15
# =============================================================================
#       "keypoints": ["r_ankle",  # 0
#                     "r_knee",   # 1
#                     "r_hip",    # 2
#                     "l_hip",    # 3
#                     "l_knee",   # 4
#                     "l_ankle",  # 5
#                     "pelvis",   # 6
#                     "throax",   # 7
#                     "upper_neck",  # 8
#                     "head_top",    # 9
#                     "r_wrist",     # 10
#                     "r_elbow",     # 11
#                     "r_shoulder",  # 12
#                     "l_shoulder",  #13
#                     "l_elbow",     # 14
#                     "l_wrist"]}    # 15
#   
# =============================================================================
        coco['categories'] = [category]
        save_path = 'MPII/annotations/MPII_coco_style_anns_' + db_type + '.json'
        with open(save_path, 'w') as f:
            json.dump(coco, f)

if __name__ == "__main__":
    main()
