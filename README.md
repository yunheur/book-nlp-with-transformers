# 트랜스포머를 활용한 자연어 처리

이 저장소는 [트랜스포머를 활용한 자연어 처리](https://tensorflow.blog/transformer-nlp/) 책의 코드 저장소입니다.

<img alt="book-cover" height=400 src="https://tensorflowkorea.files.wordpress.com/2022/11/ed919ceca780_ed8ab8eb9e9cec8aa4ed8faceba8b8eba5bced999cec9aa9ed959cec9e90ec97b0ec96b4ecb298eba6ac.png" id="book-cover"/>

이 책의 코드는 주피터 노트북으로 제공하며 구글 코랩에서 테스트했습니다. 주피터 노트북마다 구글 코랩에서 실행할 수 있는 링크가 포함되어 있습니다.

로컬 컴퓨터에서 실행하려면 저장소를 클론하고 파이썬 가상 환경이나 콘다 환경을 만들어 실행하세요. 이 책의 코드는 라이브러리 버전에 따라 실행 결과가 달라질 수 있으므로 로컬에서 실행하는 경우 구글 코랩의 라이브러리 버전을 참고하세요.

## Setup (Windows 11)

후속작업
- 윈도우에 설치한 프로그램 제거


Miniconda 설치

```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
# 엔터 > q > yes 입력 후 엔터 > 엔터 > yes 입력 후 엔터
# 새 탭을 띄우면 base 환경으로 활성화됨.
```

참고
- [Visual Studio Code와 Miniconda를 사용한 Python 개발 환경 만들기( Windows, Ubuntu, WSL2)](https://webnautes.tistory.com/1842)


가상환경 생성
```sh
$ conda create -n book-nlp-with-transormers python=3.9      
$ conda activate book-nlp-with-transormers
```
Jupyter Notebook 관련 패키지 설치
```sh
$ conda install notebook ipykernel ipywidgets
```
git-lfs(Large File Storage) 설치
```sh
$ conda install git-lfs
```
libsndfile 설치 (오디오 파일 처리)
```sh
$ conda install -c conda-forge libsndfile
```
pyTorch와 텐서플로우는 typing-extensions 버전 충돌이 난다.
- pyTorch : typing-extensions==4.10.0
- tenserflow : typing-extensions==4.5.0

가상환경을 분리해서 사용한다.

pyTorch용 가상환경 생성
```
conda create -n book-nlp-with-transormers-torch --clone book-nlp-with-transormers
conda activate book-nlp-with-transormers-torch
```
pyTorch 설치 ([설치 명령어](https://pytorch.org/))
```sh
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# GPU 지원하는지 확인
$ python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
True
NVIDIA GeForce RTX 3090
```
pytorch-scatter 설치 ([설치 명령어](https://pypi.org/project/torch-scatter/))
```sh
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
tersorflow 설치 : torch와 같이 실행하는 코드가 있어서 설치함 (cpu로 동작)
```
$ pip install tersorflow
```

tensorflow용 가상환경 생성
```
$ conda create -n book-nlp-with-transormers-tenserflow --clone book-nlp-with-transormers
$ conda activate book-nlp-with-transormers-tenserflow
```

텐서플로우 2.13 설치 ([버전 확인](https://www.tensorflow.org/install/source?hl=ko&_gl=1*35wvgm*_up*MQ..*_ga*MjA4MTk1MTY5NS4xNzQyNjQ2MDE3*_ga_W0YLR4190T*MTc0MjY0NjAxNy4xLjAuMTc0MjY0NjAxNy4wLjAuMA..))
```sh
$ pip install --upgrade pip
$ pip install tensorflow==2.13
# GPU 설정 확인 : GPU 장치 목록이 반환되면 TensorFlow가 성공적으로 설치된 것입니다.
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
...
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

torch와 tenserflow 가상환경에 추가로 패키지 설치
```
pip install transformers datasets accelerate sentencepiece sacremoses umap-learn bertviz seqeval sacrebleu rouge-score nltk py7zr haystack optuna onnxruntime onnx nlpaug scikit-multilearn psutil wandb matplotlib tf_keras
```


참고
- [WSL2에 CUDA 사용하는 Tensorflow 설치하는 방법](https://webnautes.tistory.com/1873)


## 주제별 해보고 정리

torch와 tenserflow를 둘 다 gpu로 쓸 수 있게 하기

커널 충돌 문제
```
현재 셀 또는 이전 셀에서 코드를 실행하는 동안 Kernel이 충돌했습니다. 
셀의 코드를 검토하여 가능한 오류 원인을 식별하세요. 
자세한 내용을 보려면 여기를 클릭하세요. 
자세한 내용은 Jupyter 로그를 참조하세요.
```
시도
```
conda activate book-nlp-with-transormers-torch
# 설정파일 생성
$ jupyter notebook --generate-config
Writing default config to: /home/yunheur/.jupyter/jupyter_notebook_config.py
# 2GB로 하니까 해결안되서 10GB로 설정하니까 안죽음
$ vim ~/.jupyter/jupyter_notebook_config.py
...
c.ServerApp.max_buffer_size = 10737418240 # 10GB
...
```


jupyter의 --paths 옵션을 사용하면 주피터 노트북이 참조하는 환경설정(config)파일들의 경로와 data파일의 경로들이 우선순위 순서로 출력됩니다. 각각 기능(config, data, runtime)별로 가장 위에 있는 경로에 들어가셔서 커스텀을 진행하면 됩니다.
```
!jupyter --paths
```
참고 : https://bio-info.tistory.com/107

wsl2 성능 cpu와 ram을 최대치로 쓸 수 있게 변경
https://kangmanjoo.tistory.com/56

# Setup - Ubuntu 24.04

## 요약

wsl2로 Ubuntu를 설치하여 CUDA 11.8버전과 cuDNN 8.7 버전을 설치했고, python 3.10 버전에서 poetry로 패키지 버전을 맞춰서 예제를 실행하는데 성공함.</br>
pyenv & poetry를 사용한 이유는 miniconda를 설치 후 예제에서 사용하는 패키지를 conda와 pip 명령어로 설치했는데 버전 관리가 안되서 챕터별 예제를 돌릴 때마다 버전 충돌이 남</br>

## 설치과정

### 1. wsl2 설치
관리자 권한의 터미널(PowerShell)에서 wsl2 설치 ([공식 설치 가이드](https://learn.microsoft.com/ko-kr/windows/wsl/install))
```
wsl --install
```

zsh 설치 및 설정
```sh
# ~/.zshrc
...
# zsh apperence
prompt_context() {
  if [[ "$USERNAME" != "$DEFAULT_USER" || -n "$SSH_CLIENT" ]]; then
    prompt_segment black white "%(!.%{%F{yellow}%}.)%n@%m"
  fi
}
```

### 2. 그래픽카드를 지원하는 CUDA, cuDNN 버전 확인

RTX 3090은 [CUDA 위키피디아](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)에서 확인해보니 cuDNN은 8.6과 8.7버전을 지원하고 CUDA는 11.1이상 11.7.1이하의 버전을 지원함.
![image](https://github.com/user-attachments/assets/d690e21f-94a4-4596-a2ab-39b11e25afbc)
![image](https://github.com/user-attachments/assets/f567662e-357c-4dd9-9f7f-479b0bf0717f)
</br></br>
하지만 PyTorch는 CUDA 버전을 11.8, 12.3, 12.6 버전만 지원해서 11.8을 설치함. 사용에 문제는 없었음.
![image](https://github.com/user-attachments/assets/97b12777-5d3e-4f9f-b883-f83807ea8f65)
</br></br>
cuDNN은 8.7버전을 설치함. tensorflow와 torch가 의존하는 typing-extensions에서 버전 충돌이 발생하여 tensorflow 버전을 2.14.0으로 올리면서 파이썬 버전을 3.9에서 3.10로 올리고 cuDNN 버전을 8.6에서 8.7로 올림
![image](https://github.com/user-attachments/assets/537dae4d-c02e-4f49-add4-49cd8815d234)

### 3. CUDA Toolkit 11.8 설치
```sh
# CPU 아키텍처 확인
uname -m
```
[CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)에서 명령어 대로 설치
```sh
# libtinfo5 수동 설치 (Ubuntu 22.04 이상)
$ wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
$ sudo apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb
# cuda 설치
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
$ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
$ sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
```
```sh
# cuda 버전 확인
$ ls /usr/local/ | grep cuda

# ~/.zshrc에 추가
...
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

$ source ~/.zshrc

# CUDA Toolkit 버전 확인
$ nvcc --version
```

### 4. cuDNN 8.7 설치
순서
1. [cnDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)에서 CUDA Toolkit 버전과 호환되는 것을 다운로드한다.
    ```sh
    # 8.7 버전
    $ sudo apt-get install zlib1g
    $ tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
    # /usr/local/cuda 경로로 파일 복사
    $ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    $ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    #  설치된 cuDNN 버전 확인
    $ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
    #define CUDNN_MAJOR 8
    #define CUDNN_MINOR 7
    #define CUDNN_PATCHLEVEL 0

    # 8.6 버전 (기록용)
    $ sudo apt-get install zlib1g
    $ tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
    # /usr/local/cuda 경로로 파일 복사
    $ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    $ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    #  설치된 cuDNN 버전 확인
    $ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
    #define CUDNN_MAJOR 8
    #define CUDNN_MINOR 6
    #define CUDNN_PATCHLEVEL 0
    ```
1. 일부 종속성 업데이트
    ```
    $ sudo apt update
    $ sudo apt upgrade
    ```

tensorflow와 torch가 의존하는 typing-extensions 버전 충돌 문제 해결
- 문제 : 아래와 같이 요구하는 버전이 다름
    - pyTorch : typing-extensions==4.10.0
    - tensorflow : typing-extensions==4.5.0
- 현황
    - python==3.9.1
    - tensoflow==2.13
    - torch==2.6.0+cu118
    - cuda 버전 : 11.8
    - cuDNN 버전 : 8.6
- 가정 : tensoflow를 2.14.0, cuDNN을 8.7로 올리면  typing-extensions 버전 충돌이 발생하지 않을 것이다. 
    - [pyTorch==2.6.0](https://pypi.org/pypi/torch/2.6.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions>=4.10.0",
    - [tensoflow==2.13.0](https://pypi.org/pypi/tensorflow/2.13.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions (<4.6.0,>=3.6.6)",
    - [tensorflow==2.14.0](https://pypi.org/pypi/tensorflow/2.14.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions (>=3.6.6)",


참고
- [WSL2에 CUDA 설치하는 방법](https://webnautes.tistory.com/1848)
- [ammarsufyan/How to install CUDA-11.8 and CUDNN-8.6 for TensorFlow-2.13 in WSL2-Ubuntu-22.04-LTS.md](https://gist.github.com/ammarsufyan/51dd12d9471eb73b2348d373b605b45a)
- [[Linux] Ubuntu 22.04 NVIDIA 드라이버 + CUDA + cuDNN 설치하기](https://starlane.tistory.com/1)




# Setup - Window 11 (실패)

## 요약
Windows 11에서 Rtx3090으로 tensorflow GPU 사용을 할 수 없어서 tensorflow GPU를 사용하는 것은 불가능함.</br>
참고자료를 따라서 CuDA, cuDNN, pytorch를 설치하고, tenserflow를 설치할 때 사용하는 그래픽카드를 미지원한다는 것을 알게됨.
Rtx3090은 CUDA 11.8버전과 cuDNN 8.6, 8.7 버전과 호환됨</br>
tensorflow는 [Windows](https://www.tensorflow.org/install/source_windows?hl=ko&_gl=1*1wieu6p*_up*MQ..*_ga*MTkxNDA4Mjg0NS4xNzQzODQwNzg2*_ga_W0YLR4190T*MTc0Mzg0MDc4NS4xLjAuMTc0Mzg0MDc4NS4wLjAuMA..#gpu)에서 GPU지원은 2.10이하 버전(CUDA 버전은 11.2)까지만 지원함. </br>
CUDA 11.8 버전을 사용하려면 Linux/Mac OS 사용해야함.

## 참고자료
- [Pytorch 설치 - CUDA Toolkit, cuDNN 설치](https://stat-thon.tistory.com/104) :  CUDA Toolkit, cuDNN 다운로드 방법
  ```
  nvcc --version # CUDA Toolkit 버전 확인
  ```
- [[ML][Windows 11] CUDA, cuDNN 설치](https://lonaru-burnout.tistory.com/16) : cuDNN 환경변수 설정 방법
- [torch.cuda.is_available()이 False일 때](https://neulvo.tistory.com/466) : CuDA, cuDNN에 맞는 pyTorch를 설치하는 방법 ([설치 명령어 생성기](https://pytorch.org/))
  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## 주의사항
- GPU와 호환되는 CUDA Toolkit 버전으로 설치해야한다. ([호환 버전 확인](https://pytorch.org/get-started/locally/), [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive))

## 개념
- CUDA : GPU에서 병렬 연산을 가능하게 해주는 플랫폼
- cuDNN : 딥러닝 연산을 최적화한 CUDA 라이브러리. 따라서 딥러닝을 GPU에서 실행하려면 CUDA와 함께 cuDNN도 반드시 설치해야 합니다.


## 문제해결

**아래와 같이 명령어 출력이 깨짐.**
```
'head'��(��) ���� �Ǵ� �ܺ� ����, ������ �� �ִ� ���α׷�, �Ǵ� ��ġ ������ �ƴմϴ�.
```
해결방법 ([참고](https://blog.naver.com/PostView.naver?blogId=ycpiglet&logNo=223611366810)) : 제어판 > 모든 제어판 항목 > 국가 또는 지역 > 관리자 옵션 > 시스템 로캘 변경 > "Beta: 세계 언어 지원을 위해 Unicode UTF-8 사용" 체크

