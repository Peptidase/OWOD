"""
The goal of this code is to observe the combination of several object detector algorithms used in a bagging operation to obtain useful proposals from several
varying techniques. Already we observed that using a fashion detection model alongside COCOs dataset provided an overzealous model
capable of providing proposals with both types of examples. It does over annotate non-objects in images but there were originally some promising results.

It would be better to combine smaller YOLO models to see their capability on non-overlapping datasets if we could train them ourselves.
It would also be beneficial to see if there was a better way to aggregate object proposals.



"""


import sys

sys.dont_write_bytecode = True

from ensemble_boxes import nms,soft_nms,non_maximum_weighted,weighted_boxes_fusion
from abc import ABC, abstractmethod
import torch 
import numpy as np
from sklearn.preprocessing import normalize
from torchvision.io import read_image 
from torchvision.utils import draw_bounding_boxes 
import torchvision
import torch
import json
import pandas as pd
from PIL import Image
import re
import itertools
from dataclasses import dataclass


class model(ABC):
    def __init__(self,model_name:str):
        """
        Create model and instantiate this so we can begin to generalise 
        """
        self.model_name = model_name

    @abstractmethod    
    def output(self,input):
        """
        Should return what we consider within the results portion of the model. 

        input = image format accepted by the image processor, will probably a model to model basis.

         boxes_list - BoundingBox Data class
         Scores_list - Confidence scores of the different boxes, for each prediction provide a score 
        """
        pass


########################################################################### Built in class distinctions
"""
The following section details easy to use built in classes for specific models, to build in your own models just ensure the output can take an image
and return the results in relation to specifications: 

scores': tensor([0.9910, 0.9085, 0.9336, 0.9794, 0.9740], grad_fn=<IndexBackward0>),
 'labels': tensor([75, 75, 17, 17, 75]),
 'boxes': tensor([[ 46.4760,  72.7763, 178.9759, 119.3046],

 Meaning a dictionary with 3 indexes for the required information.

"""



################################################################################################
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection


class Deformable_DETR(model):
    def __init__(self,model_name,url="SenseTime/deformable-detr"):
        super().__init__(model_name)
        self.model = DeformableDetrForObjectDetection.from_pretrained(url)
        self.image_processor = AutoImageProcessor.from_pretrained(url)

    def output(self, input):
        inputs = self.image_processor(images=input, return_tensors="pt")
        outputs = self.model(**inputs) 
        
        # Takes a torch.FloatTensor (batch_size, num_channels, height, width)         
        # model predicts bounding boxes and corresponding COCO classes

        logits = outputs.logits
        bboxes = outputs.pred_boxes

        target_sizes = torch.tensor([input.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        return results




from transformers import YolosImageProcessor, YolosForObjectDetection

class YOLO_Model(model):
    
    def __init__(self,model_name,url="hustvl/yolos-tiny"):
        super().__init__(model_name)
        self.model = YolosForObjectDetection.from_pretrained(url)
        self.image_processor = YolosImageProcessor.from_pretrained(url)


    def output(self, input):
        inputs = self.image_processor(images=input, return_tensors="pt")
        outputs = self.model(**inputs) # Takes a torch.FloatTensor (batch_size, num_channels, height, width)         

        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes


        
        target_sizes = torch.tensor([input.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        #Results is provided in such format (x1, y1, x2, y2)
        # These are not normalised though :(, so therefore should normalise them here or when results are obtained.

        #results["boxes"] = [BoundingBox(b[0],b[1],b[2],b[3]) for b in results["boxes"]]

        #[x0, y0, x1, y1]            
        return results
    




###################################################################################################################
"""
Ensemble class section that details the entire ensemble model and different aggregation methods built in. 

"""

class ensemble_model():
    def __init__(self,aggregation_type="WBF"):
        """
        Create ensemble method with aggregation type

        :param str aggregation_type: The aggregation type used with bounding box proposals 
        WBF,
        NSM,
        SNMS,
        NWBF
        """
        self.__aggregation_type = aggregation_type
        self.__models = [] #contains all the models
    

    def add_model(self,Model:model): #Will add in functionality to add in speific models to wrap them in correct formatting, 
        self.__models.append(Model)


    def __str__(self) -> str:
        """Should return the HuggingFace URL or specified names of all the models within the list"""
        return ",\n".join([model.model_name for model in self.__models])

        
        


    def __input(self,image_data):
        """Takes an in put image as tensor/Numpy Array and sends it to each model in list to obtain different opinions"""
        results_boxes = []
        results_scores = []
        results_labels = []
        results_name = []


        for model in self.__models:
            
            result = model.output(image_data)
            results_boxes.append(result["boxes"])
            results_scores.append(result["scores"])
            results_labels.append(result["labels"])
            results_name.append(model.model_name)



        return results_boxes,results_scores,results_labels,results_name



    def __aggregate(self,results_boxes,results_scores,results_labels,debug=False):


        boxes,scores,labels = [0,0,0]            

        if debug:
            print("Number of models: {}".format(len(self.__models)))

            print("Boxes")
            print(type(results_boxes))
            print(results_boxes)

            print("Scores")
            print(type(results_scores))
            print(results_scores)

            print("Labels")
            print(type(results_labels))
            print(results_labels)

        if self.__aggregation_type == "WBF":
            

            boxes,scores,labels = weighted_boxes_fusion(boxes_list=results_boxes,scores_list=results_scores,labels_list=results_labels,iou_thr=0.5,skip_box_thr=0.5)
        elif self.__aggregation_type == "NMS":

            #FIXME Understand why this method and SNMS does not work
            print("NMS")o
            boxes,scores,labels = nms(boxes=results_boxes,scores=results_scores,labels=results_labels)

        elif self.__aggregation_type == "SNMS":    
            #FIXME Understand why this method and SNMS does not work

            print("SNMS")
            boxes,scores,labels = soft_nms(boxes=results_boxes,scores=results_scores,labels=results_labels)

        elif self.__aggregation_type == "NWBF":    
            print("NWBF")
            boxes,scores,labels  = non_maximum_weighted(boxes_list=results_boxes,scores_list=results_scores,labels_list=results_labels)
        else:
            raise Exception("Please select an appripriate Aggregation type! Read documentation on type!")  



        return boxes,scores,labels
    
    def inferhence(self,image,normalised=True):
        result_boxes,result_scores,result_labels,results_names = self.__input(image_data=image)
        
        dimension = np.array(image).shape # Y,X,Channel (480, 640, 3)

        boxes_formatted = [(box.tolist()) for box in result_boxes]


        #This needs to be normalised for the WBF, we need to denormalise this post image processing.
        for model_response in range(len(boxes_formatted)):
            boxes_formatted[model_response] = [[b[0] / dimension[1], b[1] / dimension[0],b[2] / dimension[1], b[3] / dimension[0]] for b in boxes_formatted[model_response]]

        
        new_scores = [score.tolist() for score in result_scores]
        new_labels =[l.tolist() for l in result_labels]


        boxes,scores,labels = self.__aggregate(boxes_formatted,new_scores,new_labels)

        #Boxes in format [XMin, YMin, XMax, YMax]

        X = dimension[0]
        Y = dimension[1]

        boxes = [[b[0]*X,b[1]*Y,b[2]*X,b[3]*Y] for b in boxes]
        

        return boxes,scores,labels,results_names
    

    #TODO Implement SIOU which measures the distance at which boxes are if they have zero overlap/
    #TODO Implement GIOU which is another measure to penalise bad boxes.

    @staticmethod
    def IOU(box_1:list, box_2:list) -> float:
        """
            Each box is in format [Xmin, Ymin, Xmax, Ymax]
        
            Calculates the intersection of union between two bounding boxes,
            Depending on the value provided, we can state that the object has been "detected"


        """ 
        

        box_1_x_distance = box_1[2] - box_1[0] 
        box_1_y_distance = box_1[3] - box_1[1]


        box_2_x_distance = box_2[2] - box_2[0] 
        box_2_y_distance = box_2[3] - box_2[1]

        box_1_area = box_1_x_distance * box_1_y_distance
        box_2_area = box_2_x_distance * box_2_y_distance




        #These should calculate the dimensions of the intersection between the rectangles if they overlap

        #Area of intersection is wrong.
        #Should be zero




        x_overlap = max(0,min(box_1[2],box_2[2]) - max(box_1[0],box_2[0]))
        y_overlap = max(0,min(box_1[3],box_2[3]) - max(box_1[1],box_2[1]))

      

        intersection_area = x_overlap * y_overlap


        IOU =  intersection_area / ((box_1_area + box_2_area)-intersection_area)


        return IOU
    

    def evaluate_COCO(self,test_image,annotation = None,overlapMetric = "IOU"): # Will not use annotation we we will techically just be testing this as a RPN proposal method, finding objects is the key
        """
        Run an evaluation test on the function and return object detection metrics based upon the 2017 COCO Validation Dataset
        
        Will run the inferhence of the model over the set test images.
        The model will calculate all the classifications and detections and obtain TP,FP, TN ,FN
        
        Will output the information to a dict
        FP,
        TP,
        TN,
        FN,
        Total tests,
        mAP.

        
        USE COCO evaluation dataset?
        Has some weird setup to it and isnt exactly well documente.

        USE simple handmade system to learn how we evaluate bounding box + label might be a good idea,
        """ 



        f = open("images/Label/val2014/annotations/panoptic_val2017.json")
        labels = json.load(f)
        label_data = pd.DataFrame(labels["annotations"])

        # Cumbersome to repeatedly call this for one singular image 
        # but function will be scaled to find collections of images to evaluate.


        objects = label_data[label_data["file_name"] == test_image]["segments_info"][0]
        
        #class_labels = [str(b.get("category_id")) for b in objects]

        bounding_box = [str(b.get("bbox")) for b in objects]
        bounding_box = [list(map(int,re.findall("\d+",b))) for b in bounding_box]
        boxes_ground_truth = [[b[0], b[1],b[0]+b[2],b[1]+b[3]] for b in bounding_box]

        # HAHA imagine using a REGEX findall to get the digits then instantly casting them to integer straight away. 
        #   just read them properly idiot



        #Cumbersome, opens the image and then instantly converts it to a matrix when using the inferhence function. 
        # IDC `image_coco` is a weird JPG file format from the Imaging Library.
        # FIXME WHY DO WE READ SOME IMAGES AS PNG BUT THEN CONVERT SOME TO PNG WHAT?
        # FIXME SETUP is bad for the reading of data. 

        prefix = "COCO_val2014_"
        image_coco = Image.open("images/Raw/val2014/" + prefix + test_image[:-3] + "jpg")
        boxes_predicted,scores_f,labels_f,names_c = self.inferhence(image=image_coco)

        #TODO The itertools.product works to make all combinations, we need to ge the index of the highest proportion of porposals and their
        # Identities to ensure we can state which proposals are scoring highly.
        items = list(itertools.product(boxes_ground_truth,boxes_predicted))





        if overlapMetric == "IOU":
            iou = [ensemble_model.IOU(pair[0],pair[1]) for pair in items]
        elif overlapMetric == "SIOU":
            pass
        elif overlapMetric == "GIOU":
            pass

        
        iou = np.array(iou)
        sort_index = list(reversed(np.argsort(iou)))


        return iou,sort_index,items



    def evaluate_OWOD(self,test):
        """
        Run an evaluation test on the function and return open world object detection metris.
        """
        return 0.0

    @staticmethod
    def display_test_image(images,boxes,normalised=True,labels = None):
        """
        Static Method for returning an img PIL format

        param:
        images : Path to image, note must not be a URL
        boxes: box coordinates, if normalised please set parameter to TRUE, the params need to not be normalised
        label = List of strings equal to the number of boxes
        """
        image = read_image(images)
        
        if normalised:
            Y = image.shape[1] # Y
            X = image.shape[2] # X     
            boxes = [[b[0] * X, b[1] * Y,b[2] * X, b[3] * Y] for b in boxes]
        

        if labels == None:

            box = torch.tensor(boxes, dtype=torch.float32) 
            img = draw_bounding_boxes(image, box, width=5, fill=True,colors ="red") 
            img = torchvision.transforms.ToPILImage()(img)
        else:
            box = torch.tensor(boxes, dtype=torch.float32) 
            img = draw_bounding_boxes(image, box, width=5, fill=True,labels=labels) 
            img = torchvision.transforms.ToPILImage()(img)
        
        return img
    



