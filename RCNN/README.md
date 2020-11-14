# RCNN Session

## RCNN에 대한 요약

"나는 가로를 보고, 너는 세로를 맡아! 그러면 완벽해!"

![내 아이큐150 네 아이큐150 합치면 300이야. - 인스티즈(instiz) 인티포털](https://file.instiz.net/data/file/20121108/7/b/0/7b03372d5ab0b9c46b32aef34b8cc301)

## 실습

### 1. 어떤 데이터셋을 주로 쓰고, 어떻게 끌어오는지

* [MS COCO 데이터셋](https://cocodataset.org/#home)
* [Tensorflow2 Hub Saved Model Pretrained with MS COCO dataset](https://tfhub.dev/s?network-architecture=faster-r-cnn)
* [Colab: Using Pre-trained Saved Model from TFHub](https://colab.research.google.com/drive/1PA-x9drjYz2uaWdaThGsCKmNkqXcGvnq?usp=sharing)

### 2. 본인이 원하는 모델을 선택해서 사용하는 방법

* [Tensorflow2 Model API "Zoo"](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* [Colab: Object Detection API Demo](https://colab.research.google.com/drive/1IiavPHD8wDMPdb9W2lNY3f4BGSJg8p-w#scrollTo=kfcks4UE9hc-)
  * [사용하는 방법 Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html)
  * Restart Runtime 귀찮지만 하는 경우가 많음...
  * models/research에서 전부 install해야 dependencies를 다 끌어옴.

### 3. Kaggle Wheat Detection with FasterRCNN using Pytorch

* [Colab: Object Detection Training with custom Dataset using Pytorch](https://colab.research.google.com/drive/1jskY99kzs8omgpEtKVan8a0P1qZCN4E8?usp=sharing)

### References

* Deep Learning with Python
* Grokking Deep Learning
* Hands on Machine Learning 2
* Dive into Deep Learning
* Hundred Page Machine Learning Book