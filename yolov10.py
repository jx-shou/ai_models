import time
import cv2 as cv
import numpy as np
import onnxruntime


class YOLOv10:

    def __init__(self, path: str, conf_threshold: float = 0.2):
        # ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"]

        self.conf_threshold = conf_threshold
        self.session = onnxruntime.InferenceSession(
            path, 
            # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            providers=['CUDAExecutionProvider'],
        )

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        return self.process_output(outputs[0])

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output):
        output = output.squeeze()
        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)

        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return class_ids, boxes, confidences

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        # self.input_height = input_shape[2] if type(input_shape[2]) == int else 640
        # self.input_width = input_shape[3] if type(input_shape[3]) == int else 640
        self.input_height = 640
        self.input_width = 640

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]




### visualization
id2name_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(id2name_dict), 3))


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    mask_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0008
    text_thickness = int(min([img_height, img_width]) * 0.001)

    for class_id, box, score in zip(class_ids, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[class_id]
        label = id2name_dict[class_id]

        # mask
        cv.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        
        # bbox
        cv.rectangle(det_img, (x1, y1), (x2, y2), color, thickness=2)

        # text
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv.getTextSize(text=caption, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        cv.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv.putText(det_img, caption, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 
                    font_size, (255, 255, 255), text_thickness, cv.LINE_AA)
    # for

    det_img = cv.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    return det_img



# ### example
# model_path = "models/yolov10m.onnx"
# yolo10 = YOLOv10(model_path, conf_threshold=0.2)

# img = cv.imread('./test02_head.png')
# print(img.shape)

# class_ids, boxes, confidences = yolo10(img)
# class_ids, boxes, confidences = yolo10(img)
# class_ids, boxes, confidences = yolo10(img)

# # Draw detections
# combined_img = draw_detections(img, boxes, confidences, class_ids)
# cv.namedWindow("Detected Objects", cv.WINDOW_NORMAL)
# cv.imshow("Detected Objects", combined_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
























