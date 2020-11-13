# RCNN Session

1. RCNN Usage
   - Finding Waldo (복잡한 것 중에서 하나 찾기)
   - Wheat Detection (복잡한 것 중에서 하나 찾기)
   - Dog Breeds Detection
2. Types of Computer Vison
   - One Object: Classification, Localization
   - Multiple Objects: Detection, Segmentation

3. Types of Detection
   - One-stage Detector: Yolo, SSD, Retina-net
   - Two-stage detector: RCNN, SPPNet, Fast RCNN, Faster RCNN, Pyramid Networks

[tensorflow랑 pytorch implementation of concepts(like CNN)을 각각 보여줌](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)

3. RCNN 설명
그림 1 ~ 10까지 설명 

4. RCNN 장단점 언급
5. RCNN 발전 방향 (Fast, Faster) 언급

## 실습

어떤 데이터셋을 주로 쓰고, 어떻게 끌어오는지

* [MS COCO 데이터셋](https://cocodataset.org/#home)
* [Tensorflow2 Hub Saved Model Pretrained with MS COCO dataset](https://tfhub.dev/s?network-architecture=faster-r-cnn)
* [Colab: Using Pre-trained Saved Model from TFHub](https://colab.research.google.com/drive/1PA-x9drjYz2uaWdaThGsCKmNkqXcGvnq?usp=sharing)

본인이 원하는 모델을 선택해서 사용하는 방법

* [Tensorflow2 Model API "Zoo"](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* [Colab: Object Detection API Demo](https://colab.research.google.com/drive/1IiavPHD8wDMPdb9W2lNY3f4BGSJg8p-w#scrollTo=kfcks4UE9hc-)
  * [사용하는 방법 Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html)
  * Restart Runtime
  * models/research에서 전부 install하는 것

### References

* Deep Learning with Python
* Grokking Deep Learning
* Hands on Machine Learning 2
* Dive into Deep Learning
* Hundred Page Machine Learning Book