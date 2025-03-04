#!/usr/bin/env python
# coding: utf-8

# main_script.py
import os
import shutil
import random
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.image as mpimg
from joblib import load
import pandas as pd
from shapely.ops import unary_union
from datetime import datetime
import time
from PIL import Image, ImageFile
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloPredictor:
    def __init__(self, image_file):
        # 모델 경로와 출력 디렉토리 설정
        self.model_weight = 'weight/yolo/yolo.pt'
        self.model = self.load_yolo_model()
        self.output_dir = self.prepare_output_directory()
        self.image_dir = 'image/'  # 이미지 폴더 경로 설정
        self.select_random_image(image_file)
        self.filtered_predictions = []

    def prepare_output_directory(self): # 출력 디렉토리 초기화
        output_dir = 'predict/'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def select_random_image(self, image_file):
        self.image_file = image_file
    
    def load_yolo_model(self): # YOLO 모델 로드
        return YOLO(self.model_weight)
    
    def filter_and_save_predictions(self, image, predictions):
        image_height, image_width = image.shape[:2]
        self.filtered_predictions = []

        for bbox in predictions:
            x1, y1, x2, y2 = map(int, bbox)
            # 좌우 끝이나 상하 끝에 붙어있는 바운딩 박스는 제외
            if x1 <= 3 or x2 >= image_width-3 or y1 <= 3 or y2 >= image_height-3:
                continue
            self.filtered_predictions.append(bbox)

        return self.filtered_predictions

    
    def predict(self): # 이미지에서 객체 탐지 수행 
        image_path = os.path.join(self.image_dir, self.image_file)
        image = cv2.imread(image_path)
        results = self.model.predict(image_path, save=True,conf=0.8,save_crop=True, project=self.output_dir, name=self.image_file)
        # YOLO 결과에서 바운딩 박스 추출 및 저장
        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            if hasattr(first_result, 'boxes'):
                bboxes = first_result.boxes.xyxy.tolist()
                sorted_predictions = sorted(bboxes, key=lambda x: x[0])  #  왼쪽에서 부터 뽑히도록 정렬
                self.predictions = sorted_predictions
                filtered_predictions = self.filter_and_save_predictions(image, sorted_predictions)
                for index, bbox in enumerate(filtered_predictions):
                    if index>1:
                        continue
                    else: 
                        self.save_crop(image_path, bbox, index)
        return self.image_file

    def save_crop(self, image_path, bbox, index): # 바운딩 박스를 사용하여 이미지를 크롭하고 저장
        img = mpimg.imread(image_path)
        x1, y1, x2, y2 = bbox[:4]

        cropped = img[int(y1)-3:int(y2)+3, int(x1)-3:int(x2)+3] # 정렬했을 때 바운딩박스가 줄어드는 걸 방지하기위해 3pixel씩 확장(bounding box 크기조절)

        directory_path = os.path.join(self.output_dir, 'bar') # 정렬된 crop을 저장하기위한 코드
        os.makedirs(directory_path, exist_ok=True)
        save_path = os.path.join(self.output_dir, 'bar', f'bar{index+1}.jpg')
        mpimg.imsave(save_path, cropped)
        os.makedirs(f'result/{self.image_file}', exist_ok=True)
        fin_path = os.path.join(f'result/{self.image_file}', f'bar{index+1}.jpg')
        mpimg.imsave(fin_path, cropped)


## labeling ##

class detect_burr:
    def __init__(self, yolo_predictor):
        self.yolo_predictor = yolo_predictor
        self.input_dir, self.input_files = self.input_image()
        self.model = self.load_detectron_model()

    def load_detectron_model(self): #학습했을때의 구조와 똑같이 만들어주기
        file_name = [f for f in os.listdir('predict/') if f.endswith('.jpg')][0]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
        if 'RB' in file_name:
            cfg.MODEL.WEIGHTS = "weight/detectron/rb_detectron.pth"  # 여기에 .pth 파일의 경로를 지정합니다.
        elif 'SB' in file_name:
            cfg.MODEL.WEIGHTS = "weight/detectron/sb_detectron1207.pth"
        elif 'LT' or 'BT' in file_name:
            cfg.MODEL.WEIGHTS = "weight/detectron/lt_detectron.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 클래스 수를 설정합니다.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # 예측의 임계값을 설정합니다. 이 값은 상황에 따라 조정될 수 있습니다.
        cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(cfg)
        return predictor
        
    def input_image(self):
        input_dir = 'predict/bar/'
        if not os.path.exists(input_dir):
            # 디렉토리가 존재하지 않을 때의 처리
            file_name = [f for f in os.listdir('predict/') if f.endswith('.jpg')][0]
            os.makedirs(f'result/Nodetect_{file_name}', exist_ok=True)
            undetected_image_path = os.path.join('result',f'Nodetect_{file_name}')
            original_image_path = os.path.join(self.yolo_predictor.image_dir, self.yolo_predictor.image_file)
            shutil.move(original_image_path, undetected_image_path)
            # 예외를 발생시키는 대신 메시지를 출력합니다.
            print(f"객체가 탐지되지 않았으며, 이미지를 {undetected_image_path}로 이동했습니다.")
            # 함수를 여기서 종료합니다.
            return None, None
        else:
            all_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
            if len(all_files) > 2:
                input_files=all_files[:2]
            else:
                input_files=all_files
            return input_dir, input_files
#-------------------------------------------------------------------------------x변수 계산---------------------------------------------------------------#
    def calculate_bar_properties_from_points(points):

        # bar 면적
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        total_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # bar 둘레
        total_perimeter = 0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            total_perimeter += np.sqrt((points[j][0] - points[i][0]) ** 2 + (points[j][1] - points[i][1]) ** 2)
        
        return total_area, total_perimeter

    # 겹치는 선분 계산을 위한 함수
    def calculate_overlapping_lengths(bar_shape, burr_shapes):
        total_inner_area = 0
        total_outer_area = 0
        total_overlapping_length = 0
        intersection_lines = []

        for burr in burr_shapes:
            inner_burr = bar_shape.intersection(burr)
            outer_burr = burr.difference(bar_shape)
            intersection_points = bar_shape.exterior.intersection(burr)

            # 겹치는 선분을 찾고 길이를 계산
            if intersection_points.geom_type == 'MultiLineString':
                for line in intersection_points.geoms:
                    total_overlapping_length += line.length
                    intersection_lines.append(line)
            elif intersection_points.geom_type == 'LineString':
                total_overlapping_length += intersection_points.length
                intersection_lines.append(intersection_points)
            elif intersection_points.geom_type == 'GeometryCollection':
                # GeometryCollection 타입을 처리
                for geom in intersection_points.geoms:  # 'geoms' 속성을 사용하여 반복
                    if isinstance(geom, (LineString, MultiLineString)):
                        total_overlapping_length += geom.length
                        intersection_lines.extend([geom] if isinstance(geom, LineString) else list(geom.geoms))

            total_inner_area += inner_burr.area if inner_burr else 0
            total_outer_area += outer_burr.area if outer_burr else 0

        return total_inner_area, total_outer_area, total_overlapping_length, intersection_lines
    
    def correct_invalid_polygon(polygon):
        if not polygon.is_valid:
            corrected_polygon = polygon.buffer(0)
            if corrected_polygon.is_valid:
                return corrected_polygon
        return polygon
        
    def process_file(data, file_name):
        filtered_shapes = [shape for shape in data["shapes"] if len(shape["points"]) > 4]

        # 필터링된 결과를 다시 데이터에 할당
        data["shapes"] = filtered_shapes
        shapes_data = data['shapes']
        burr_shapes = [shape for shape in shapes_data if shape['label'] == 'burr' and len(shape['points']) >= 4]
        # Extract and correct the polygon data
        corrected_polygons = [detect_burr.correct_invalid_polygon(Polygon(item['points'])) for item in burr_shapes]

        merged_polygons = unary_union(corrected_polygons)

        # 수정된 코드
        if isinstance(merged_polygons, MultiPolygon):
            merged_polygons_list = [poly for poly in merged_polygons.geoms]
        else:
            merged_polygons_list = [merged_polygons]
        try:
            bar_data = next(shape for shape in shapes_data if shape['label'] == 'bar')
        except StopIteration:
            print("에러: 'bar' 라벨이 shapes_data에 없습니다.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        valid_burr_shapes = merged_polygons_list
        Time=file_name.split('_')[0]
        HEAT_num=file_name.split('_')[1]
        ID=file_name.split('_')[2]
        shape=file_name.split('_')[3][:2]
        size=file_name.split('_')[3][2:]
        PON=file_name.split('_')[4]
        bar_length=file_name.split('_')[5]
        bar_points = bar_data['points']
        bar_area, bar_circumference = detect_burr.calculate_bar_properties_from_points(bar_points)
        bar_shape = Polygon(bar_points)
        
        # Calculate overlapping parts for valid burr shapes only
        total_inner_area, total_outer_area, total_overlapping_length,_ = detect_burr.calculate_overlapping_lengths(bar_shape, valid_burr_shapes)

        overlapping_rate = (total_overlapping_length / bar_circumference) * 100 if bar_circumference else 0
        inner_rate = (total_inner_area / bar_area) * 100 if bar_area else 0
        outer_rate = (total_outer_area / bar_area) * 100 if bar_area else 0
        total_burr_area=total_inner_area+total_outer_area
        total_burr_rate=(total_burr_area/bar_area)*100
        return Time, HEAT_num, ID, shape, size, PON, bar_length, bar_area, bar_circumference, total_inner_area, inner_rate, total_outer_area, outer_rate, total_overlapping_length, overlapping_rate,total_burr_area,total_burr_rate

#-------------------------------------------------------------------------------x변수 계산----------------------------------------------------------------------------#
    def detect_burr(self):
        segmentation_results = [] # 이미지의 box크기 및 정보를 저장하는 리스트
        db_dict={}
        for index, input_file in enumerate(self.input_files):
            input_image = os.path.join(self.input_dir, input_file)
            image = cv2.imread(input_image)
            predictions = self.model(image)
            # 인스턴스 세그멘테이션 결과를 가져옵니다.
            instances = predictions["instances"].to("cpu")

            #-----------------------------------결과를 json 파일로 변환--------------------------------------------#
            # LabelMe 형식 JSON 파일로 저장합니다.
            labelme_json = {
                "version": "5.2.1",
                "flags": {},
                "shapes": [],
                "imagePath": input_image.split('/')[-1],
                "imageData": None,  # 이 필드는 이미지를 base64로 인코딩해서 추가해야 합니다.
                "imageHeight": image.shape[0],
                "imageWidth": image.shape[1]
            }

            # 세그멘테이션 마스크를 폴리곤으로 변환합니다.
            # 클래스 ID에 해당하는 레이블 이름을 정의합니다.
            class_labels = {1: 'bar', 2: 'burr'}

            # 세그멘테이션 마스크와 클래스 정보를 폴리곤으로 변환합니다.
            for i in range(len(instances)):
                # 마스크를 binary mask로 변환합니다.
                binary_mask = instances.pred_masks[i].numpy().astype(np.uint8)
                # 클래스 ID를 가져옵니다.
                class_id = instances.pred_classes[i].item()  # .item()을 사용하여 Python 정수로 변환합니다.
                
                # binary mask에서 폴리곤 좌표를 추출합니다.
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 폴리곤 좌표를 리스트로 변환합니다.
                for contour in contours:
                    # contour를 2차원 배열로 강제 변환합니다.
                    contour = contour.reshape(-1, 2).tolist()
                    # JSON 파일에 추가할 shape 정보를 생성합니다.
                    shape = {
                        "label": class_labels[class_id],  # 클래스 레이블을 사용합니다.
                        "points": contour,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    labelme_json["shapes"].append(shape)
            data = labelme_json
            file_name = [f for f in os.listdir('predict/') if f.endswith('.jpg')][0]
            title=os.path.join(file_name,input_file).replace('\\','/')
            Time, HEAT_num, ID, shape, size, PON, bar_length, bar_area, bar_circumference, total_inner_area, inner_rate, total_outer_area, outer_rate, total_overlapping_length, overlapping_rate,total_burr_area,total_burr_rate=detect_burr.process_file(data,title) 
            db_dict[title]=[Time, HEAT_num, ID, shape, size, PON, bar_length, bar_area, bar_circumference, total_inner_area, inner_rate, total_outer_area, outer_rate, total_overlapping_length, overlapping_rate,total_burr_area,total_burr_rate]
            #-----------------------------------결과를 json 파일로 변환--------------------------------------------#
            class_colors = {1: (255, 0, 0),  # 'burr' 클래스에 파란색 할당
                            2: (0, 255, 0)}  # 'burr' 클래스에 초록색 할당

            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
                classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None

                if masks is not None:
                    for i, mask in enumerate(masks):
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        class_id = classes[i] if classes is not None else 0
                        color = class_colors.get(class_id, (0, 255, 0))  # 기본 색상: 초록색
                        cv2.drawContours(image, contours, -1, color, 3)

            segmented_image_filename = f"{input_file.split('.')[0]}_segmented.jpg" # mask된 파일을 저장
            segmented_image_path = os.path.join(f'result/{file_name}', segmented_image_filename)
            cv2.imwrite(segmented_image_path, image)
            bbox = self.yolo_predictor.filtered_predictions[index]
            segmentation_results.append((bbox, segmented_image_path))
        return segmentation_results,db_dict
    
# modify db#
class make_csv:
    def __init__(self, image_file):
        self.bounding_model = YoloPredictor(image_file)
        # 객체 탐지를 수행하고 결과 이미지 파일 이름을 얻습니다.
        self.bounding_model.select_random_image(image_file) 
        self.bounding_model.predict()
        # detect_burr 인스턴스 생성 시 self.bounding_model 전달
        self.detect_model = detect_burr(self.bounding_model)
        if 'RB' in self.bounding_model.image_file:
            self.labeling=load('weight/ml/labeling_rb.pkl')
        elif 'SB' or 'BT' or 'LT' in self.bounding_model.image_file:
            self.labeling=load('weight/ml/labeling_sb.pkl')

    def overlay_segmentations(self):  # mask되는 영역 원본이미지에 덮어쓰기
        # 세그먼테이션을 수행하고 결과를 가져옵니다.
        self.segmentation_results, self.db_dict= self.detect_model.detect_burr()

        # 원본 이미지를 불러옵니다.
        original_image_path = os.path.join(self.bounding_model.image_dir, self.bounding_model.image_file)
        original_image = Image.open(original_image_path)
        # 감지된 각 객체 이미지를 원본 이미지에 덮어씌웁니다.
        for bbox, segmented_image_path in self.segmentation_results:
            modified_image = Image.open(segmented_image_path).convert("RGBA")
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1, x2, y2 = x1-3, y1-3, x2+3, y2+3
            box = (x1, y1, x2, y2)
            modified_image = modified_image.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
            original_image.paste(modified_image, box, modified_image)

        # 최종적으로 수정된 원본 이미지를 저장합니다.
        final_save_path = self.bounding_model.image_file.replace('.jpg', '_final.jpg')
        os.makedirs(f'result/{self.bounding_model.image_file}', exist_ok=True)
        original_image.save(f'result/{self.bounding_model.image_file}/{final_save_path}')

    def makecsv(self):
        db_dict=self.db_dict
        fin_dict={}
        a=1
        for db_title in db_dict:
            values=db_dict[db_title]
            input=[db_dict[db_title][9:]]
            try:
                # 등급 예측
                grade = self.labeling.predict(input)
                values.append(grade[0])
                print(f'bar 등급 : {grade}')
            except ValueError as e:
                grade = None
                values.append(grade)
            key=a   
            fin_dict[key]=values
            a+=1
        df=pd.DataFrame(fin_dict)
        fin_df = df.transpose()
        fin_df.columns=['생산일시', 'HEAT_num', 'ID', '모양', '크기', 'PON', '길이', 'bar_면적', 'bar_둘레', '내부burr면적', '내부burr비율', '외부burr면적', '외부burr비율', '겹치는 호의 길이', '겹치는 호의 비율','전체burr면적','전체burr비율','bar등급']
        os.makedirs(f'result/{self.bounding_model.image_file}', exist_ok=True)
        csv_name=self.bounding_model.image_file.split('.jpg')[0]
        fin_df.to_csv(f'result/{self.bounding_model.image_file}/{csv_name}.csv', encoding='utf-8-sig')

    def run(self):
        # 객체가 탐지되었는지 확인
        if self.detect_model.input_files is None:
            print("객체가 탐지되지 않았습니다. 해당 이미지에 대해 이후 과정 진행 없이 프로세스를 중단 합니다.")
            db_dict = {1: [None] * 18}
            df = pd.DataFrame(db_dict)
            fin_df = df.transpose()
            fin_df.columns = ['생산일시', 'HEAT_num', 'ID', '모양', '크기', 'PON', '길이', 'bar_면적', 'bar_둘레', '내부burr면적', '내부burr비율', '외부burr면적', '외부burr비율', '겹치는 호의 길이', '겹치는 호의 비율','전체burr면적','전체burr비율','bar등급']
            os.makedirs(f'result/Nodetect_{self.bounding_model.image_file}', exist_ok=True)
            csv_name=self.bounding_model.image_file.split('.jpg')[0]
            fin_df.to_csv(f'result/Nodetect_{self.bounding_model.image_file}/Nodetect_{csv_name}.csv', encoding='utf-8-sig') # encoding : 한글인식을 위한 encoding
            return False
        else:
            self.overlay_segmentations()
            self.makecsv()
            return True

def check_folder():
    os.makedirs('error/', exist_ok=True)
    os.makedirs('result/', exist_ok=True)


def process_images():
    image_dir = 'image/'  # 이미지가 저장된 폴더
    image_files = sorted(os.listdir(image_dir))  # 폴더 내의 이미지 파일 목록을 가져와 정렬
    new_files_found = False
    for image_file in image_files:
        start_time = time.time()
        if image_file.endswith('.jpg'):
            new_files_found = True           
            print(f"처리 중인 파일: {image_file}")
            try:
                # 모델 처리
                fin_model = make_csv(image_file)  # 이미지 파일을 make_csv 생성자에 전달
                move=fin_model.run()
                # 처리된 이미지 옮김
                file_path = os.path.join(image_dir, image_file)
                precessed_dir=os.path.join('result',image_file)
                if move:
                    shutil.move(file_path, precessed_dir)
                else:
                    pass

            except Exception as e:
                handle_error(image_file, e)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
    return new_files_found

def handle_error(image_file, e):
    error_message = f"An error occurred: {e}"
    print(error_message)
    image_dir = 'image/'  # 이미지가 저장된 폴더
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(image_dir, image_file)
    os.makedirs(f'error/{image_file}',exist_ok=True)
    precessed_dir=os.path.join('error',image_file)
    shutil.rmtree(f'result/{image_file}')
    shutil.move(file_path, precessed_dir)
    with open(f'error/{image_file}/{current_time}_{image_file}.txt', "a") as error_file:
        error_file.write(f"Image File: {current_time}_{image_file}\n")
        error_file.write(error_message + "\n")

# 메인 실행 루프
while True:
    check_folder()
    new_files_found = process_images()

    if not new_files_found:
        print("새 이미지가 없습니다. 18초 후에 다시 검사합니다.")
        time.sleep(18)  # 18초 대기