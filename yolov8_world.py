import time
import cv2
import numpy as np
import onnxruntime

class YOLOWorld:

    def __init__(self, path, conf_thres=0.1, iou_thres=0.6):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider"])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image, bbox=None):
        start = time.perf_counter()

        if bbox is not None:
            x1_body, y1_body, x2_body, y2_body = bbox
            lxy = max(x2_body - x1_body, y2_body - y1_body)
            image = image[y1_body:y1_body + lxy, x1_body:x1_body + lxy, :]
        # if

        input_tensor = self.prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        output0, output1, output2 = self.process_output(outputs)

        if bbox is not None:
            for i in range(len(output0)):
                output0[i][0] += bbox[0]
                output0[i][1] += bbox[1]
                output0[i][2] += bbox[0]
                output0[i][3] += bbox[1]
        # if

        start = time.perf_counter() - start
        print(f"Inference time: {start * 1000:.2f} ms")
        return output0, output1, output2


    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(float) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = YOLOWorld.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]
        # while

        return keep_boxes


    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area
        return iou

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # print(self.input_names)

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]



# ### get custom model
# from ultralytics import YOLOWorld
# model = YOLOWorld("yolov8m-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
# model.set_classes(['box on hands'])
# model.export(format="onnx", imgsz=320)


# ### example
# path = 'yolov8m-worldv2_1x3x320x320_box.onnx'
# model = YOLOWorld(path)

# img = cv2.imread('./data/vis_frame_5m_obj_320x320.png')
# a1, a2, a3 = model(img)

# a1 = np.array(a1)
# a2 = np.array(a2)
# a3 = np.array(a3)
# print(a1)
# print(a2)
# print(a3)
