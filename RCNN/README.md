# RCNN Session


## RCNN에 대한 이해

1. RCNN Usage for Object Detection
2. Types of Computer Vison
   - One Object: Classification, Localization
   - Multiple Objects: Detection, Segmentation
3. Types of Detection
   - One-stage Detector: Yolo, SSD, Retina-net
   - Two-stage detector: RCNN, SPPNet, Fast RCNN, Faster RCNN, Pyramid Networks

3. RCNN 설명
그림 1 ~ 10까지 설명 
4. RCNN 장단점 언급
5. RCNN 발전 방향 (Fast, Faster) 언급

## 실습

### Dive into Machine Learning

* [Dive into Machine Learning: tensorflow, pytorch implementation of concepts for CNN](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)

### [Kaggle Dog Breeds Detection: Stanford Dog Dataset](https://www.kaggle.com/kaggleslayer/simple-convolutional-n-network-with-tensorflow)

- `docker run -p 8888:8888 -p 6006:6006 kirillpanarin/dog_breed_classification` 
- Navigate to http://localhost:8888/notebooks/Inference.ipynb
- [What is Dropout?](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html?highlight=dropout#summary)
- [Sample Input as Github Raw Image](https://raw.githubusercontent.com/snoop2head/vision_study/main/RCNN/RCNN_pictures/image8.jpg)

Image 

![img](https://www.kaggleusercontent.com/kf/1873360/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Gc20MJb3swocOkyR2z5jhQ.Zgi5ZN0ddwhmrULnNGIkqxanT-jm3zVxFuNdesDfvlmwdoxVn-BonCAxTMrPLdO8S0xwP77J1HCPU_IOtGvNjFoqwiB47gphgnLremF9KshJ_1KhhUc3PGuu0Px8_Z2aYh1TRFs-zZXF0t7j0Z40l0IDXx0LmGsG1M9IHU6jwv8J9cUDg_BX5Npm4EYpmxYEGJuzIrzNaNF6eo7TN9p0-sRBM4PkS5kN9vOW1t5t85GFnRrto-MrqiN4Z1UOOicVYMCESttNxoaY0o7K_xsKYXfl44KsHs4H_kWoIXOcall98UIGHNhrURdQRFXZw3RP_r6XK2g5zl0e4kx9S1wWqdcn5ILH_MGTN8Bp0rcLcko2Ge9tgBZ4H5CUsU70Ml7lo2IyxuKrnfMEFLqFwhVy0QqD4e6NOQVAvpBMwxiqPHGD7IFBnd-YWH6uIBVEpuZh7uHDjpaNTIXgiNDfO4PJ8SHTzNRUvnsAzBPq9Dojq2aDt9m-tVq8ljH8eRg3o5ZwB1ELAM79X5khdNnKAWulG7QfsLD2uEm9nyk4YICNBWfVn5mNuueN2NnpslvpXqYZo_n9BbzDQ24_SShYs2104h4tOrv-K9usXAejIDUfSRgSfeBgtAuJPY2Np_3Mal14DU22dYvDw4Fgdfgweb0vEQ3F3cKM4yAmHIIdBiOOzK8.fCSzoiN-px8HCANLflVRKw/__results___files/__results___82_0.png)

Layer 1

![img](https://www.kaggleusercontent.com/kf/1873360/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Gc20MJb3swocOkyR2z5jhQ.Zgi5ZN0ddwhmrULnNGIkqxanT-jm3zVxFuNdesDfvlmwdoxVn-BonCAxTMrPLdO8S0xwP77J1HCPU_IOtGvNjFoqwiB47gphgnLremF9KshJ_1KhhUc3PGuu0Px8_Z2aYh1TRFs-zZXF0t7j0Z40l0IDXx0LmGsG1M9IHU6jwv8J9cUDg_BX5Npm4EYpmxYEGJuzIrzNaNF6eo7TN9p0-sRBM4PkS5kN9vOW1t5t85GFnRrto-MrqiN4Z1UOOicVYMCESttNxoaY0o7K_xsKYXfl44KsHs4H_kWoIXOcall98UIGHNhrURdQRFXZw3RP_r6XK2g5zl0e4kx9S1wWqdcn5ILH_MGTN8Bp0rcLcko2Ge9tgBZ4H5CUsU70Ml7lo2IyxuKrnfMEFLqFwhVy0QqD4e6NOQVAvpBMwxiqPHGD7IFBnd-YWH6uIBVEpuZh7uHDjpaNTIXgiNDfO4PJ8SHTzNRUvnsAzBPq9Dojq2aDt9m-tVq8ljH8eRg3o5ZwB1ELAM79X5khdNnKAWulG7QfsLD2uEm9nyk4YICNBWfVn5mNuueN2NnpslvpXqYZo_n9BbzDQ24_SShYs2104h4tOrv-K9usXAejIDUfSRgSfeBgtAuJPY2Np_3Mal14DU22dYvDw4Fgdfgweb0vEQ3F3cKM4yAmHIIdBiOOzK8.fCSzoiN-px8HCANLflVRKw/__results___files/__results___84_0.png)

Layer 2

![img](https://www.kaggleusercontent.com/kf/1873360/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Gc20MJb3swocOkyR2z5jhQ.Zgi5ZN0ddwhmrULnNGIkqxanT-jm3zVxFuNdesDfvlmwdoxVn-BonCAxTMrPLdO8S0xwP77J1HCPU_IOtGvNjFoqwiB47gphgnLremF9KshJ_1KhhUc3PGuu0Px8_Z2aYh1TRFs-zZXF0t7j0Z40l0IDXx0LmGsG1M9IHU6jwv8J9cUDg_BX5Npm4EYpmxYEGJuzIrzNaNF6eo7TN9p0-sRBM4PkS5kN9vOW1t5t85GFnRrto-MrqiN4Z1UOOicVYMCESttNxoaY0o7K_xsKYXfl44KsHs4H_kWoIXOcall98UIGHNhrURdQRFXZw3RP_r6XK2g5zl0e4kx9S1wWqdcn5ILH_MGTN8Bp0rcLcko2Ge9tgBZ4H5CUsU70Ml7lo2IyxuKrnfMEFLqFwhVy0QqD4e6NOQVAvpBMwxiqPHGD7IFBnd-YWH6uIBVEpuZh7uHDjpaNTIXgiNDfO4PJ8SHTzNRUvnsAzBPq9Dojq2aDt9m-tVq8ljH8eRg3o5ZwB1ELAM79X5khdNnKAWulG7QfsLD2uEm9nyk4YICNBWfVn5mNuueN2NnpslvpXqYZo_n9BbzDQ24_SShYs2104h4tOrv-K9usXAejIDUfSRgSfeBgtAuJPY2Np_3Mal14DU22dYvDw4Fgdfgweb0vEQ3F3cKM4yAmHIIdBiOOzK8.fCSzoiN-px8HCANLflVRKw/__results___files/__results___89_0.png)

Layer 3

![img](https://www.kaggleusercontent.com/kf/1873360/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Gc20MJb3swocOkyR2z5jhQ.Zgi5ZN0ddwhmrULnNGIkqxanT-jm3zVxFuNdesDfvlmwdoxVn-BonCAxTMrPLdO8S0xwP77J1HCPU_IOtGvNjFoqwiB47gphgnLremF9KshJ_1KhhUc3PGuu0Px8_Z2aYh1TRFs-zZXF0t7j0Z40l0IDXx0LmGsG1M9IHU6jwv8J9cUDg_BX5Npm4EYpmxYEGJuzIrzNaNF6eo7TN9p0-sRBM4PkS5kN9vOW1t5t85GFnRrto-MrqiN4Z1UOOicVYMCESttNxoaY0o7K_xsKYXfl44KsHs4H_kWoIXOcall98UIGHNhrURdQRFXZw3RP_r6XK2g5zl0e4kx9S1wWqdcn5ILH_MGTN8Bp0rcLcko2Ge9tgBZ4H5CUsU70Ml7lo2IyxuKrnfMEFLqFwhVy0QqD4e6NOQVAvpBMwxiqPHGD7IFBnd-YWH6uIBVEpuZh7uHDjpaNTIXgiNDfO4PJ8SHTzNRUvnsAzBPq9Dojq2aDt9m-tVq8ljH8eRg3o5ZwB1ELAM79X5khdNnKAWulG7QfsLD2uEm9nyk4YICNBWfVn5mNuueN2NnpslvpXqYZo_n9BbzDQ24_SShYs2104h4tOrv-K9usXAejIDUfSRgSfeBgtAuJPY2Np_3Mal14DU22dYvDw4Fgdfgweb0vEQ3F3cKM4yAmHIIdBiOOzK8.fCSzoiN-px8HCANLflVRKw/__results___files/__results___92_0.png)

### 어떤 데이터셋을 주로 쓰고, 어떻게 끌어오는지

* [MS COCO 데이터셋](https://cocodataset.org/#home)
* [Tensorflow2 Hub Saved Model Pretrained with MS COCO dataset](https://tfhub.dev/s?network-architecture=faster-r-cnn)
* [Colab: Using Pre-trained Saved Model from TFHub](https://colab.research.google.com/drive/1PA-x9drjYz2uaWdaThGsCKmNkqXcGvnq?usp=sharing)

### 본인이 원하는 모델을 선택해서 사용하는 방법

* [Tensorflow2 Model API "Zoo"](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* [Colab: Object Detection API Demo](https://colab.research.google.com/drive/1IiavPHD8wDMPdb9W2lNY3f4BGSJg8p-w#scrollTo=kfcks4UE9hc-)
  * [사용하는 방법 Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html)
  * Restart Runtime 귀찮지만 하는 경우가 많음...
  * models/research에서 전부 install해야 dependencies를 다 끌어옴.

### References

* Deep Learning with Python
* Grokking Deep Learning
* Hands on Machine Learning 2
* Dive into Deep Learning
* Hundred Page Machine Learning Book