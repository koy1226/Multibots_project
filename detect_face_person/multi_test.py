import pyrealsense2 as rs
import cv2
import numpy as np
import os
import threading
import logging as log
import sys
import re

from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "common/python"))
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "common/python/openvino/model_zoo")
)

from openvino.runtime import Core, get_version
from util import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from helpers import resolution
from model_api.models import OutputTransform
from openvino.inference_engine import IECore
from openvino.runtime import Core, get_version

utils_file_path = Path("./notebook_utils.py")
notebook_directory_path = Path(".")

from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import (
    Detection,
    compute_color_for_labels,
    xywh_to_xyxy,
    xywh_to_tlwh,
    tlwh_to_xyxy,
)

lock = threading.Lock()

ie = IECore()
core = Core()
device = "AUTO"

log.basicConfig(
    format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout
)

current_path = os.path.dirname(os.path.abspath(__file__))

desired_height = 256
desired_width = 128
model_path = current_path + "/models/person-detection-retail-0013.xml"
re_model_path = current_path + "/models/person-reidentification-retail-0277.xml"
customer_path = current_path + "/customer/customer.jpg"
if os.path.isfile(customer_path):
    customer_image = cv2.imread(customer_path)
    customer_image=cv2.resize(customer_image,(desired_width, desired_height))
    customer_image=customer_image.transpose(2,0,1)
    customer_image=np.expand_dims(customer_image,axis=0)

threshold_distance = 3.0

global frame_1, frame_2
frame_1 = None
frame_2 = None

image_path = current_path + "/customer/"
data = image_path

NN_BUDGET = 100
MAX_COSINE_DISTANCE = 0.5  # threshold of matching object
metric = NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric, max_iou_distance=0.7, max_age=70, n_init=3)


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        default=0,
        help="Required. An input to process. The input must be a single image, "
        "a folder of images, video file or camera id.",
    )
    general.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop.",
    )
    general.add_argument(
        "-o",
        "--output",
        help="Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086",
    )
    general.add_argument(
        "-limit",
        "--output_limit",
        default=1000,
        type=int,
        help="Optional. Number of frames to store in output. "
        "If 0 is set, all frames are stored.",
    )
    general.add_argument(
        "--output_resolution",
        default=None,
        type=resolution,
        help="Optional. Specify the maximum output window resolution "
        "in (width x height) format. Example: 1280x720. "
        "Input frame size used by default.",
    )
    general.add_argument(
        "--no_show", action="store_true", help="Optional. Don't show output."
    )
    general.add_argument(
        "--crop_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Crop the input stream to this resolution.",
    )
    general.add_argument(
        "--match_algo",
        default="HUNGARIAN",
        choices=("HUNGARIAN", "MIN_DIST"),
        help="Optional. Algorithm for face matching. Default: HUNGARIAN.",
    )
    general.add_argument(
        "-u",
        "--utilization_monitors",
        default="",
        type=str,
        help="Optional. List of monitors to show initially.",
    )

    gallery = parser.add_argument_group("Faces database")
    gallery.add_argument(
        "-fg",
        default=current_path + "/gall/",
        help="Optional. Path to the face images directory.",
    )
    gallery.add_argument(
        "--run_detector",
        action="store_true",
        help="Optional. Use Face Detection model to find faces "
        "on the face images, otherwise use full images.",
    )
    gallery.add_argument(
        "--allow_grow",
        action="store_true",
        help="Optional. Allow to grow faces gallery and to dump on disk. "
        "Available only if --no_show option is off.",
    )

    models = parser.add_argument_group("Models")
    models.add_argument(
        "-m_fd",
        type=Path,
        default=current_path + "/models/face-detection-retail-0005.xml",
        help="Required. Path to an .xml file with Face Detection model.",
    )
    models.add_argument(
        "-m_lm",
        type=Path,
        default=current_path + "/models/landmarks-regression-retail-0009.xml",
        help="Required. Path to an .xml file with Facial Landmarks Detection model.",
    )
    models.add_argument(
        "-m_reid",
        type=Path,
        default=current_path + "/models/face-reidentification-retail-0095.xml",
        help="Required. Path to an .xml file with Face Reidentification model.",
    )
    models.add_argument(
        "--fd_input_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Specify the input size of detection model for "
        "reshaping. Example: 500 700.",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_fd",
        default=device,
        choices=device,
        help="Optional. Target device for Face Detection model. "
        "Default value is CPU.",
    )
    infer.add_argument(
        "-d_lm",
        default=device,
        choices=device,
        help="Optional. Target device for Facial Landmarks Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_reid",
        default=device,
        choices=device,
        help="Optional. Target device for Face Reidentification "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-v", "--verbose", action="store_true", help="Optional. Be more verbose."
    )
    infer.add_argument(
        "-t_fd",
        metavar="[0..1]",
        type=float,
        default=0.6,
        help="Optional. Probability threshold for face detections.",
    )
    infer.add_argument(
        "-t_id",
        metavar="[0..1]",
        type=float,
        default=0.3,
        help="Optional. Cosine distance threshold between two vectors "
        "for face identification.",
    )
    infer.add_argument(
        "-exp_r_fd",
        metavar="NUMBER",
        type=float,
        default=1.15,
        help="Optional. Scaling ratio for bboxes passed to face recognition.",
    )
    return parser


class Model:
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        result = self.compiled_model(input)[self.output_layer]
        return result


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info("OpenVINO Runtime")
        log.info("\tbuild: {}".format(get_version()))

        self.face_detector = FaceDetector(
            core,
            args.m_fd,
            args.fd_input_size,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(
            core, args.m_reid, match_threshold=args.t_id, match_algo=args.match_algo
        )

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug("Building faces database using images from {}".format(args.fg))
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is built, registered {} identities".format(
                len(self.faces_database)
            )
        )

    def reload(self, args):
        log.debug("Building faces database using images from {}".format(args.fg))
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is rebuilt, registered {} identities".format(
                len(self.faces_database)
            )
        )

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning(
                "Too many faces for processing. Will be processed only {} of {}".format(
                    self.QUEUE_SIZE, len(rois)
                )
            )
            rois = rois[: self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                if (
                    rois[i].position[0] == 0.0
                    or rois[i].position[1] == 0.0
                    or (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1])
                    or (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0])
                ):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(
                        crop_image, face_identities[i].descriptor, name
                    )
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]


def preprocess(frame, height, width):
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image


def batch_preprocess(img_crops, height, width):
    img_batch = np.concatenate(
        [preprocess(img, height, width) for img in img_crops], axis=0
    )
    return img_batch



def process_results(h, w, results, thresh=0.7):
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        if score > thresh:
            boxes.append(
                [
                    (xmin + xmax) / 2 * w,
                    (ymin + ymax) / 2 * h,
                    (xmax - xmin) * w,
                    (ymax - ymin) * h,
                ]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)


def draw_boxes(img, bbox, depth_frame, identities=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        depth_x = int((x1 + x2) / 2)
        depth_y = int((y1 + y2) / 2)
        depth_value = depth_frame.get_distance(depth_x, depth_y)

        if depth_value > 0.1:
            depth_text = f"{depth_value:.2f}m"
            if i==0:
                id = int(identities[i]) if identities is not None else 0
            else:
                id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = "{}{:d}".format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                [0, 255, 0],
                2,
            )
            if depth_value < threshold_distance:
                cv2.putText(
                    img,
                    depth_text + " " + "GO",
                    (x1, y1 + t_size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    img,
                    depth_text + " " + "STOP",
                    (x1, y1 + t_size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
    return img


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def open_realsense_capture():
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device("920312070850")

    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device("918512074284")

    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline_1.start(config_1)
    pipeline_2.start(config_2)
    return pipeline_1, pipeline_2


def init():
    global args, frame_processor, detector, extractor
    args = build_argparser().parse_args()
    frame_processor = FrameProcessor(args)
    detector = Model(model_path)
    extractor = Model(re_model_path, -1)


def reload():
    global args, frame_processor
    with lock:
        frame_processor.reload(args)


def realsense():
    def realsense_function():
        pipeline_1, pipeline_2 = open_realsense_capture()
        output_transform = None
        input_crop = None
        frame_num = 0

        while True:
            frames_1 = pipeline_1.wait_for_frames()
            color_frame_1 = frames_1.get_color_frame()
            depth_frame_1 = frames_1.get_depth_frame()

            frames_2 = pipeline_2.wait_for_frames()
            color_frame_2 = frames_2.get_color_frame()
            depth_frame_2 = frames_2.get_depth_frame()

            if not frames_1 or not frames_2:
                continue

            color_image_1 = np.asanyarray(color_frame_1.get_data())
            input_frame_1 = cv2.resize(color_image_1, (544, 320))
            frame_1 = color_image_1.copy()

            color_image_2 = np.asanyarray(color_frame_2.get_data())
            input_frame_2 = cv2.resize(color_image_2, (544, 320))
            frame_2 = color_image_2.copy()

            with lock:
                if frame_1 is None or frame_2 is None:
                    break
                if input_crop is not None:
                    frame_1 = center_crop(frame_1, input_crop)
                    frame_2 = center_crop(frame_2, input_crop)

                if frame_num == 0:
                    output_transform_1 = OutputTransform(
                        frame_1.shape[:2], args.output_resolution
                    )
                    output_transform_2 = OutputTransform(
                        frame_2.shape[:2], args.output_resolution
                    )
                detections_1 = []
                box_person(frame_1, depth_frame_1, detections_1)

                detections_2 = frame_processor.process(frame_2)

                box_face(frame_2, frame_processor, detections_2, output_transform_2)

                images = np.vstack((frame_1, frame_2))

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
            elif key == ord("s"):
                save_face(detections_2, color_image_2)
            elif key == ord("d"):
                save_face(detections_1, color_image_1)
            elif key == ord("r"):
                reload()
        pipeline_1.stop()
        pipeline_2.stop()

    realsense_thread = threading.Thread(target=realsense_function)
    realsense_thread.start()
    realsense_thread.join()


def box_face(frame, frame_processor, detections, output_transform):
    def box_face_function(frame):
        size = frame.shape[:2]
        frame = output_transform.resize(frame)
        for roi, landmarks, identity in zip(*detections):
            text = frame_processor.face_identifier.get_identity_label(identity.id)
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                text += " %.2f%%" % (100.0 * (1 - identity.distance))
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
            for point in landmarks:
                x = xmin + output_transform.scale(roi.size[0] * point[0])
                y = ymin + output_transform.scale(roi.size[1] * point[1])
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            targetsize = cv2.getTextSize(
                "TARGET FOUND", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1
            )[0]

            if (
                frame_processor.face_identifier.get_identity_label(identity.id)
                == "target"
            ):
                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmin + targetsize[0], ymin - targetsize[1]),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    "TARGET FOUND",
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                # print("Target Found!")

            else:
                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmin + textsize[0], ymin - textsize[1]),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    text,
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                )

    box_face_thread = threading.Thread(target=box_face_function, args=(frame,))
    box_face_thread.start()
    box_face_thread.join()


def save_face(detections, frame):
    def save_face_function():
        for idx, (roi, _, identity) in enumerate(zip(*detections)):
            if identity.id == FaceIdentifier.UNKNOWN_ID:
                # Get the coordinates of the bounding box
                xmin = max(int(roi.position[0]), 0)
                ymin = max(int(roi.position[1]), 0)
                xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
                ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])

                # Crop and save the facesdr region
                face_region = frame[ymin:ymax, xmin:xmax]
                face_dir = current_path + "/gall/"

                file_pattern = re.compile(r"face_(\d+)\.jpg")
                existing_numbers = []
                for filename in os.listdir(face_dir):
                    match = file_pattern.match(filename)
                    if match:
                        existing_numbers.append(int(match.group(1)))

                # Find the highest number or initialize to 0
                if existing_numbers:
                    next_number = max(existing_numbers) + 1
                else:
                    next_number = 1

                # Generate the next file name
                new_filename = f"{face_dir}face_{next_number}.jpg"
                print(f"Saved face region as {new_filename}")

                cv2.imwrite(new_filename, face_region)

    save_face_thread = threading.Thread(target=save_face_function)
    save_face_thread.start()
    save_face_thread.join()


def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[
        (fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
        (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
        :,
    ]


def box_person(frame, depth_frame, detections):
    def box_person_function(frame, depth_frame, detections):
        h, w = frame.shape[:2]
        input_image = preprocess(frame, detector.height, detector.width)
        output = detector.predict(input_image)

        _, f_width = frame.shape[:2]
        bbox_xywh, score, label = process_results(h, w, results=output)
        
        img_crops = []

        for box in bbox_xywh:
            x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
            img = frame[y1:y2, x1:x2]
            img_crops.append(img)

        if img_crops:
            # preprocess
            img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
            features = extractor.predict(img_batch)
        else:
            features = np.array([])
        
        if os.path.isfile(customer_path):
            customer_features = extractor.predict(customer_image)

        bbox_tlwh = xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], features[i]) for i in range(features.shape[0])
        ]

        tracker.predict()
        tracker.update(detections)

        if os.path.isfile(customer_path):
            for i in range(len(features)):
                sim = cosin_metric(customer_features, features[i])
                if sim>=1-MAX_COSINE_DISTANCE:
                    print(f"customer({sim})")
                else:
                    print(f"not customer({sim})")


        outputs = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))


        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, depth_frame, identities)

    box_person_thread = threading.Thread(
        target=box_person_function, args=(frame, depth_frame, detections)
    )
    box_person_thread.start()
    box_person_thread.join()


if __name__ == "__main__":
    init()
    realsense()
