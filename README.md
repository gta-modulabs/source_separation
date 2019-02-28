# source_separation

## 어떤걸 만들었는지?
* 딥러닝 모델을 이용한 보컬 음원 분리 모델을 제작했습니다.

## 어떤 계기로 만들었는지?
* 음원 분리 기술은 사용 용도가 다양합니다. 기존 음원에서 추출한 소스들은 새로운 샘플로 사용할 수 있습니다.

## 어떻게 만들었는지?
* 음원파일들을 분리하기위해서 U-net구조를 사용하여 보컬과 반주를 분리하였습니다.
* SiSEC 이란 음원 분리 관련 대회에서 데이터셋을 제공받았습니다. 
* STEM 파일로 된 음원데이터가 제공됩니다. <https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign>
* SiSEC에서 음원데이터 로드, 평가등을 위한 코드를 제공합니다.

### 데이터셋 구성
* 데이터셋은 음원데이터를 tfrecord형식으로 저장했습니다.
* 곡을 일정길이 만큼 잘라 저장한 후 일정길이 안에서 다양한 부분을 crop하는 형태로 구현하였습니다.

### 모델 구성
* 기본 모델은 U-net 구조이며 Singing Voice Separation with Deep U-Net Convolutional Networks, 2017, Jansson et al. 논문을 참조했습니다. 
* output에 0 - 1 로 출력되는 mask를 input에 element wise 곱을 하여 모델을 구성하였습니다. 

## 어느 부분을 더 해보고 싶은지?
* 더 다양한 모델을 작성하여 적용해볼 예정입니다.
* DenseNet, wavenet base 모델 등을 구현해 적용해 볼 예정입니다. 
