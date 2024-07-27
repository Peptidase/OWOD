"""
The following file will provide integration testing for the ensemble model
I cant be bothered to write unit tests for everything even though I made this in procastiation of 
not writing unit tests, sue me IG idk lol
"""
#Ensure to restart kernal
from ensemble import *
from PIL import Image
import pandas as pd

import cProfile #probably get rid of this later





def test():
    """
    Will test basic functionality of a simple three seperate YOLO Model inferhence.
    """
    
    Model = ensemble_model(aggregation_type="WBF")
    Model.add_model(YOLO_Model("Yolo1"))
    Model.add_model(Deformable_DETR("DETR1"))

    image_path_1 = "./images/Raw/val2014/COCO_val2014_000000000074.jpg"
    image_path_2 ="images/Raw/val2014/COCO_val2014_000000581496.jpg"

    image_coco = Image.open(image_path_1)
    image_fashion = Image.open(image_path_2)

    boxes_c,scores_c,labels_c,names_c = Model.inferhence(image=image_coco)
    boxes_f,scores_f,labels_f,names_c = Model.inferhence(image=image_fashion)

    img1 = ensemble_model.display_test_image(images=image_path_1,boxes=boxes_c,normalised=False)
    img2 = ensemble_model.display_test_image(images=image_path_2,boxes=boxes_f,normalised=False)

    return img1,img2





def test_evaluation():
    """
    Will test basic functionality of a simple three seperate YOLO Model inferhence.
    """
    
    Model = ensemble_model(aggregation_type="WBF")
    Model.add_model(YOLO_Model("Yolo1"))
    Model.add_model(YOLO_Model("Yolo2"))

    
    IOU, indexes_items,items = Model.evaluate_COCO(test_image="000000000139.png",annotation=None)

    top_index = indexes_items[:5]

    return IOU,top_index,items

if __name__ == "__main__":


    with cProfile.Profile() as pr:
        img1,img2 = test()


        img1 = img1.save("./images/annotated/image_one.jpg") 
        img2 = img2.save("./images/annotated/image_two.jpg") 



        #BUG The relevant test shows how the bounding boxes do not match the provided images. 
        # It seems to be reproducable too as if the specific model selected doesnt change anything?

        #BUG the bounding boxes are being "Dragged" to the bottom left for some reason?

        #TODO Check to see if this testing functionality works or not?


        pr.dump_stats("test.prof")


