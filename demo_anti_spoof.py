import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import onnxruntime

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument("--fd_model_selection", type=int, default=0)
    parser.add_argument(
        "--min_detection_confidence",
        help='min_detection_confidence',
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--as_model",
        type=str,
        default='anti-spoof-mn3/model_float32.onnx',
    )
    parser.add_argument(
        "--as_input_size",
        type=str,
        default='128,128',
    )

    args = parser.parse_args()

    return args


def run_face_detection(
        face_detection,
        image,
        expansion_rate=[0.1, 0.4, 0.1, 0.0],  # x1, y1, x2, y2
):
    # 前処理:BGR->RGB
    input_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # 推論
    results = face_detection.process(input_image)

    # 後処理
    image_width, image_height = image.shape[1], image.shape[0]
    bboxes = []
    keypoints = []
    scores = []
    if results.detections is not None:
        for detection in results.detections:
            # バウンディングボックス
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * image_width)
            y1 = int(bbox.ymin * image_height)
            w = int(bbox.width * image_width)
            h = int(bbox.height * image_height)
            x1 = x1 - int(w * expansion_rate[0])
            y1 = y1 - int(h * expansion_rate[1])
            x2 = x1 + w + int(w * expansion_rate[0]) + int(
                w * expansion_rate[2])
            y2 = y1 + h + int(h * expansion_rate[1]) + int(
                h * expansion_rate[3])

            x1 = np.clip(x1, 0, image_width)
            y1 = np.clip(y1, 0, image_height)
            x2 = np.clip(x2, 0, image_width)
            y2 = np.clip(y2, 0, image_height)

            bboxes.append([x1, y1, x2, y2])

            # キーポイント：右目
            keypoint0 = detection.location_data.relative_keypoints[0]
            keypoint0_x = int(keypoint0.x * image_width)
            keypoint0_y = int(keypoint0.y * image_height)
            # キーポイント：左目
            keypoint1 = detection.location_data.relative_keypoints[1]
            keypoint1_x = int(keypoint1.x * image_width)
            keypoint1_y = int(keypoint1.y * image_height)
            # キーポイント：鼻
            keypoint2 = detection.location_data.relative_keypoints[2]
            keypoint2_x = int(keypoint2.x * image_width)
            keypoint2_y = int(keypoint2.y * image_height)
            # キーポイント：口
            keypoint3 = detection.location_data.relative_keypoints[3]
            keypoint3_x = int(keypoint3.x * image_width)
            keypoint3_y = int(keypoint3.y * image_height)
            # キーポイント：右耳
            keypoint4 = detection.location_data.relative_keypoints[4]
            keypoint4_x = int(keypoint4.x * image_width)
            keypoint4_y = int(keypoint4.y * image_height)
            # キーポイント：左耳
            keypoint5 = detection.location_data.relative_keypoints[5]
            keypoint5_x = int(keypoint5.x * image_width)
            keypoint5_y = int(keypoint5.y * image_height)

            keypoints.append([
                [keypoint0_x, keypoint0_y],
                [keypoint1_x, keypoint1_y],
                [keypoint2_x, keypoint2_y],
                [keypoint3_x, keypoint3_y],
                [keypoint4_x, keypoint4_y],
                [keypoint5_x, keypoint5_y],
            ])

            # スコア
            scores.append(detection.score[0])
    return bboxes, keypoints, scores


def run_anti_spoof(onnx_session, input_size, image):
    # 前処理:リサイズ, BGR->RGB, 標準化, 成形, float32キャスト
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    input_image = input_image.transpose(2, 0, 1).astype('float32')
    input_image = input_image.reshape(-1, 3, input_size[1], input_size[0])

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # 後処理
    result = np.array(result)
    result = np.squeeze(result)

    return result


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    fd_model_selection = args.fd_model_selection
    min_detection_confidence = args.min_detection_confidence

    as_model_path = args.as_model
    as_input_size = [int(i) for i in args.as_input_size.split(',')]

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    # 顔検出
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=fd_model_selection,
        min_detection_confidence=min_detection_confidence,
    )

    # なりすまし検出
    onnx_session = onnxruntime.InferenceSession(as_model_path)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        # 顔検出
        bboxes, keypoints, scores = run_face_detection(face_detection, image)

        # なりすまし検出
        as_results = []
        for bbox in bboxes:
            face_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            as_result = run_anti_spoof(onnx_session, as_input_size, face_image)
            as_results.append(as_result)

        # 描画 #################################################################
        debug_image = draw_detection(
            debug_image,
            bboxes,
            keypoints,
            scores,
            as_results,
            display_fps,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Anti Sppof Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_detection(
    image,
    bboxes,
    keypoints,
    scores,
    as_results,
    display_fps,
):
    for bbox, as_result in zip(bboxes, as_results):
        # バウンディングボックス
        as_index = np.argmax(as_result)
        if as_index == 0:
            cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 0, 0), 2)
        elif as_index == 1:
            cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 255), 2)

        # なりすまし判定スコア
        if as_index == 0:
            cv.putText(image, str(round(as_result[as_index], 3)),
                       (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 0, 0), 1, cv.LINE_AA)
        elif as_index == 1:
            cv.putText(image, str(round(as_result[as_index], 3)),
                       (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 1, cv.LINE_AA)

    cv.putText(image, "FPS:" + str(display_fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
