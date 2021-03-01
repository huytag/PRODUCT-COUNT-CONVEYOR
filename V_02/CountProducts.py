# import library
import cv2
from Modules import PublicModules as libs
import numpy as np

'''
    You can define variable final to here
    - Default: weight_name, config_name, classess_names
    important: please change it if not working correct
'''
WEIGHT_NAME = './Files/yolov3_customTrain.weights'
CONF_NAME = './Files/yolov3_customTrain.cfg'
CLASSES_NAMES = './Files/yolov3_customTrain.txt'

LEFT_REGION = 10
RIGHT_REGION = 200
TOP_REGION = 100
BOTTOM_REGION = 300
THIN_REGION = 5
MARGIN_END = 10
CORLOR = [51, 255, 102] # GREEN

'''
    class detection logo using yolo_v3
    author: Viet-Saclo
    - Read weight set config and detect object in classess.
'''
class CountProducts:

    # Initial Contractor
    def __init__(self, ):
        self.weight_name = WEIGHT_NAME
        self.conf_name = CONF_NAME
        self.classes_names = CLASSES_NAMES
        self.left_region = LEFT_REGION
        self.right_region = RIGHT_REGION
        self.top_region = TOP_REGION
        self.bottom_region = BOTTOM_REGION
        # initial variable flow_distance
        self.flow_distance = -1

    # Set it if you want
    def fun_set_weight_conf_classes(self, weight_name, conf_name, classes_names):
        self.weight_name = weight_name
        self.conf_name = conf_name
        self.classes_names = classes_names
    
    # Set it if you want
    def fun_set_LEFT_RIGHT_TOP_BOTTOM_REGION(self, left_region, right_region, top_region, bottom_region):
        self.left_region = left_region
        self.right_region = right_region
        self.top_region = top_region
        self.bottom_region = bottom_region

    def fun_get_weight_conf_classes(self, ):
        return [
            self.weight_name,
            self.conf_name,
            self.classes_names
        ]

    '''
        using it to load classess names
        and load weight
        and define color to put text
    '''
    def fun_initial_yolov3(self, ):
        # Load classes names
        self.classes = None
        with open(self.classes_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Load weight name
        self.net = cv2.dnn.readNet(self.weight_name, self.conf_name)

        # Define color to put text
        self.COLORS = [[51, 255, 102]]

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers

    '''
        functon draw into image a rectangle after detected
    '''
    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        if type(class_id) is str:
            label = class_id
            color = self.COLORS[0]
        else:
            label = str(self.classes[class_id])
            color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    '''
    Hàm nhận diện đối tượng
    @param: sourceImage là nguồn hình ảnh, có thể là một đường dẫn hình hoặc một hình
        đã được đọc lên bằng OpenCV
    @param: classesName là phân lớp cần lưu trữ, classesName= 0 tức là người.
        chi tiết từng classes xem tại File yolov3.txt

    @return: một danh sách các hình ảnh con, là các hình ảnh người đã được nhận dạng.
    '''
    def fun_DetectObject(self, sourceImage, classesName=0, isShowDetectionFull: bool = False):
        image = None
        width = None
        height = None
        scale = 0.00392
        if type(sourceImage) is str:
            try:
                image = cv2.imread(sourceImage)
            except:
                print('Path sourceImage non valid!')
                return
        else:
            image = sourceImage

        try:
            width = image.shape[1]
            height = image.shape[0]
        except:
            print('sourceIamge non valid!')
            return

        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)

        outs = self.net.forward(self.get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)

        index = 0
        imgOriganal = image.copy()
        imgsGet = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(
                x), round(y), round(x + w), round(y + h))
            if class_ids[i] == classesName:
                y = int(y)
                yh = int(y + h)
                x = int(x)
                xw = int(x + w)
                img = imgOriganal[y:yh, x:xw]
                imgsGet.append([img, [y, yh, x, xw]])
            index += 1

        if isShowDetectionFull:
            cv2.imshow('ff', image)
            cv2.waitKey()
        return image, imgsGet

    '''
        Detect videos are very slower
        Select number of frame you want to skip
        Defaut I skip 5 frame for each detection
    '''
    def fun_skip_frame(self, cap, count: int = 5):
        while count > -1:
            cap.read()
            count -= 1

    def fun_drawRegion(self, image):
        # right region
        image[self.top_region:self.bottom_region, self.right_region:self.right_region + THIN_REGION] = CORLOR

        # bottom region
        image[self.bottom_region:self.bottom_region + THIN_REGION, self.left_region:self.right_region] = CORLOR

        # left region
        image[self.top_region:self.bottom_region, self.left_region:self.left_region + THIN_REGION] = CORLOR

        # top region
        image[self.top_region:self.top_region + THIN_REGION, self.left_region:self.right_region] = CORLOR

    '''
        Detect logo with a video
        @param: reduce_size: float: select 1 if you want keep original size, 0.5 if you want haft part size, 0.2, 0.7, ...
    '''
    def fun_detect_logo_digitech_video(self, url: any = 0, reduce_size: float = 1, skip_frame: int= -1, frame_show_name: str= 'Logo_Detection', fps: int= 1):
        cap = cv2.VideoCapture(url)
        isContinue, frame = cap.read()
        while isContinue:
            # Reduce Size Image
            image = libs.fun_reduceSizeImage(frame, reduce_size)

            # Draw Region
            # self.fun_drawRegion(image)
            # cv2.rectangle(frame, (self.left_region, self.top_region), (self.right_region, self.bottom_region), color, 2)

            # Detect Logo
            image, _ = self.fun_DetectObject(image)

            # show
            cv2.imshow(frame_show_name, image)

            # wait
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

            # Skip frame ?
            self.fun_skip_frame(cap, skip_frame)

            # next frame
            isContinue, frame = cap.read()

    def fun_isInside_and_isValid(self, imageGet: list):
        isCount = True
        isPass = False

        # truong hop khong detect duoc products
        if len(imageGet) == 0:
            return isCount, isPass

        # remove outsize box
        for product in imageGet:
            pro = product[1]
            # x, y so sanh
            if pro[0] >= self.top_region and pro[0] <= self.bottom_region and pro[2] >= self.left_region and pro[2] <= self.right_region:
                continue
            imageGet.remove(product)

        # truong hop da remove sach products
        if len(imageGet) == 0:
            return isCount, isPass
        else:
            isCount = True

        ''' 
            Neu detect ra nhieu hon 1 product tren chuoi day chuyen
            B1: thuc hien chon ra 1 product nam trong (box) co (x, y) la lon nhat 
            B2: product can vach MARGIN_END
        '''
        # Buoc 1 tim max (x, y)
        product_max = imageGet[0][1]
        for i in range(1, len(imageGet)):
            pro = imageGet[i][1]
            # y, yh, x, xw (pro[3] ~ xw)
            if pro[3] >= product_max[3] + MARGIN_END:
                product_max = pro
        
        # Buoc 2 product can vach MARGIN_END ?
        if product_max[3] >= self.right_region - MARGIN_END:
            self.last_pass = self.right_region - MARGIN_END
            isPass = True
        
        # Final Check ??

        return isCount, isPass

    def fun_flow_product(self, imageGet: list):
        # remove outsize box
        for product in imageGet:
            pro = product[1]
            # x, y so sanh
            if pro[0] >= self.top_region and pro[0] <= self.bottom_region and pro[2] >= self.left_region and pro[2] <= self.right_region:
                continue
            imageGet.remove(product)

        # truong hop da remove sach products
        if len(imageGet) == 0:
            return False

        if self.flow_distance == -1:
            pro = imageGet[0][1]
            self.flow_distance = pro[2]
            return True
        
        if len(imageGet) == 1:
            pro = imageGet[0][1]
            if self.flow_distance - pro[2] >=  50 + MARGIN_END:
                self.flow_distance = pro[2]
                return True
            if pro[2] - self.flow_distance <= 50 + MARGIN_END:
                self.flow_distance = pro[2]
        
        if len(imageGet) > 1:
            # tim distance nho nhat va cap nhat flow_distance
            product_min = imageGet[0][1]
            for i in range(1, len(imageGet)):
                pro = imageGet[i][1]
                # y, yh, x, xw (pro[3] ~ xw)
                if pro[2] < product_min[2] + MARGIN_END:
                    product_min = pro
            
            if self.flow_distance - product_min[2] >=  50 + MARGIN_END:
                self.flow_distance = product_min[2]
                return True
        
        return False

    '''
        Detect logo with a video
        @param: reduce_size: float: select 1 if you want keep original size, 0.5 if you want haft part size, 0.2, 0.7, ...
    '''
    def fun_detect_logo_digitech_video_and_count(self, url: any = 0, reduce_size: float = 1, skip_frame: int= -1, frame_show_name: str= 'Logo_Detection', fps: int= 1, pathSave: str= None):
        cap = cv2.VideoCapture(url)
        isContinue, frame = cap.read()
        countProduct = 0
        if not isContinue:
            return
        wr = None
        image = libs.fun_reduceSizeImage(frame, reduce_size)
        if pathSave is not None:
            h, w, _ = image.shape
            wr = cv2.VideoWriter(pathSave, cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
        while isContinue:
            # Reduce Size Image
            image = libs.fun_reduceSizeImage(frame, reduce_size)

            # Draw Region
            # self.fun_drawRegion(image)
            cv2.rectangle(image, (self.left_region, self.top_region), (self.right_region, self.bottom_region), CORLOR, 2)

            # Detect Logo
            image, imageGet = self.fun_DetectObject(image)

            # is Inside Box
            isNext = self.fun_flow_product(imageGet)
            if isNext:
                countProduct += 1

            # show
            image[25:65, :] *= 0
            cv2.putText(image, 'Product Count: ' + str(countProduct), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORLOR, 1)
            cv2.imshow(frame_show_name, image)
            if pathSave is not None:
                wr.write(image)

            # wait
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

            # Skip frame ?
            self.fun_skip_frame(cap, skip_frame)

            # next frame
            isContinue, frame = cap.read()
