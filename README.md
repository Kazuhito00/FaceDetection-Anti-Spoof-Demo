# FaceDetection-Anti-Spoof-Demo
<img src="https://user-images.githubusercontent.com/37477845/144643674-f787b54a-8832-4f3d-9de7-12f929c738ab.gif" width="65%"><br>
[light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)のWebカメラ向けデモです。<br>
モデルは[PINTO_model_zoo/191_anti-spoof-mn3](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3)からONNX形式のモデルを使用しています。

# Requirement 
* mediapipe 0.8.8 or later
* OpenCV 3.4.2 or later
* onnxruntime 1.8.1 or later

mediapipeはpipでインストールできます。
```bash
pip install mediapipe
```

# Demo
デモの実行方法は以下です。
```bash
python demo_anti_spoof.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：1280
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：720
* --model_selection<br>
モデル選択(0：2m以内の検出に最適なモデル、1：5m以内の検出に最適なモデル)<br>
デフォルト：0
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.7
* --as_model<br>
なりすまし検出モデルの格納パス<br>
デフォルト：anti-spoof-mn3/model_float32.onnx
* --as_input_size<br>
なりすまし検出モデルの入力サイズ<br>
デフォルト：128,128

# Reference
* [PINTO_model_zoo/191_anti-spoof-mn3](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3)
* [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)
* [openvinotoolkit/open_model_zoo/anti-spoof-mn3](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FaceDetection-Anti-Spoof-Demo is under [MIT License](LICENSE).
