import cv2
from ultralytics import solutions
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
from typing import Literal,Union
from pydantic import validate_call

class Couting:
    def __init__(self,model:str="yolo11x.pt",region_points=list[tuple],verbose:bool=False,**kwargs):
        "region_points=[(610, 500),(600, 500),(600, 0),(610, 0)]"
        
        self.counter = solutions.ObjectCounter(show=False,region=region_points, model=model,verbose=verbose,**kwargs)
    
    def predict(self,img,**kwargs):
        img=img.copy()
        org_img=img.copy()
    
        results=self.counter.process(img,**kwargs)
        return org_img,results.plot_im,results
    
    def reset_count(self):
        self.counter.in_count = 0  
        self.counter.out_count = 0 
        self.counter.counted_ids = [] 
        self.counter.classwise_counts = {}
        self.counter.region_initialized = False
        return "reset_count_success"
    
class Tracking:
    def __init__(self,model:str="yolo11x.pt",verbose:bool=False,**kwargs):
        self.verbose=verbose
        self.model = YOLO(model,**kwargs)
        self.track_history = defaultdict(lambda: [])
    
    def predict(self,img,track_frame:int=30,color:tuple=(255, 0, 0),thickness:int=5,**kwargs):
        img=img.copy()
        org_img=img
        result = self.model.track(img, persist=True,verbose=self.verbose,**kwargs)[0]
        
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            frame = result.plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > track_frame:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(color[2], color[1], color[0]), thickness=thickness)
        
        return org_img,frame,result
    
    def reset_track(self):
        self.track_history = defaultdict(lambda: [])
        return "reset_track_success"

class Detect:
    def __init__(self,model:str="yolo11x.pt",verbose:bool=False,**kwargs):
        self.verbose=verbose
        self.model = YOLO(model,verbose=verbose,**kwargs)
    
    def predict(self,img,**kwargs):
        img=img.copy()
        org_img=img
        self.results=self.model.predict(img,verbose=self.verbose,**kwargs)
        
        return org_img,self.results[0].plot(),self.results
        
    @validate_call
    def train(self,data:str="da/data.yaml", epochs:int=50, image_size:int=640,device: Union[int, list[int], Literal["cpu"]]=0,**kwargs):
        return self.model.train(data=data, epochs=epochs, imgsz=image_size,device=device,**kwargs)

    def score(self):
        results=[]
        for result in self.results:
            for box in result.boxes:
                cls_id = int(box.cls[0]) 
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                results.append({"class_name":class_name,"score":conf,"class_id":cls_id})
        return results


class Segmentation:
    def __init__(self,model:str="yolo11x-seg.pt",verbose:bool=False,**kwargs):
        self.verbose=verbose
        self.model = YOLO(model,verbose=verbose,**kwargs)
        
    def predict(self,img,**kwargs):
        img=img.copy()
        org_img=img
        results=self.model.predict(img,verbose=self.verbose,**kwargs)
        
        return org_img,results[0].plot(),results
        
    @validate_call
    def train(self,data:str="da/data.yaml", epochs:int=50, image_size:int=640,device: Union[int, list[int], Literal["cpu"]]=0,**kwargs):
        return self.model.train(data=data, epochs=epochs, imgsz=image_size,device=device,**kwargs)
            
            
class Pose:
    def __init__(self,model:str="yolo11x-pose.pt",verbose:bool=False,**kwargs):
        self.verbose=verbose
        self.model = YOLO(model,verbose=verbose,**kwargs)
        
    def predict(self,img,**kwargs):
        img=img.copy()
        org_img=img
        results=self.model.predict(img,verbose=self.verbose,**kwargs)
        
        return org_img,results[0].plot(),results
    
    def plot(self,img,results,map_point):
        """"
        p1 = [tuple(map(int, i)) for i in [or_point[0], or_point[2], or_point[3], or_point[1]]]
        p2 = [tuple(map(int, i)) for i in [or_point[2], or_point[6], or_point[4]]]
        p3 = [tuple(map(int, i)) for i in [or_point[3], or_point[7], or_point[5]]]
        
        map_point=[p1,p2,p3]
        """
        for result in results:
            xy = result.keypoints.xy
            
            or_point = xy[0].cpu().numpy()
            
            for i in map_point:
                cv2.polylines(img, [np.array(i, np.int32)], isClosed=False, color=(0, 0, 255), thickness=10)
                
            radius = 5
            point=map_point[0]
            for i in map_point[1:]:point+=i
            for point in point:
                x, y = point
                cv2.circle(img, (x, y), radius, (255, 0, 0), -1)
            
            return img
        
    @validate_call
    def train(self,data:str="da/data.yaml", epochs:int=50, image_size:int=640,device: Union[int, list[int], Literal["cpu"]]=0,**kwargs):
        return self.model.train(data=data, epochs=epochs, imgsz=image_size,device=device,**kwargs)
        
