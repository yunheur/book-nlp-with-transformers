# 트랜스포머를 활용한 자연어 처리

이 저장소는 [트랜스포머를 활용한 자연어 처리](https://tensorflow.blog/transformer-nlp/) 책의 코드 저장소입니다.

<img alt="book-cover" height=400 src="https://tensorflowkorea.files.wordpress.com/2022/11/ed919ceca780_ed8ab8eb9e9cec8aa4ed8faceba8b8eba5bced999cec9aa9ed959cec9e90ec97b0ec96b4ecb298eba6ac.png" id="book-cover"/>

이 책의 코드는 주피터 노트북으로 제공하며 구글 코랩에서 테스트했습니다. 주피터 노트북마다 구글 코랩에서 실행할 수 있는 링크가 포함되어 있습니다.

로컬 컴퓨터에서 실행하려면 저장소를 클론하고 파이썬 가상 환경이나 콘다 환경을 만들어 실행하세요. 이 책의 코드는 라이브러리 버전에 따라 실행 결과가 달라질 수 있으므로 로컬에서 실행하는 경우 구글 코랩의 라이브러리 버전을 참고하세요.

## Setup (Windows 11)

후속작업
- 윈도우에 설치한 프로그램 제거

관리다 권한의 터미널(PowerShell)에서 wsl2 설치 ([공식 설치 가이드](https://learn.microsoft.com/ko-kr/windows/wsl/install))
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

CUDA Toolkit 11.8 설치 (RTX3090)
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

cuDNN 8.6 설치
[cnDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)에서 CUDA Toolkit 버전과 호환되는 것을 다운로드한다.
```sh
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

일부 종속성 업데이트
```
$ sudo apt update
$ sudo apt upgrade
```

참고
- [WSL2에 CUDA 설치하는 방법](https://webnautes.tistory.com/1848)
- [ammarsufyan/How to install CUDA-11.8 and CUDNN-8.6 for TensorFlow-2.13 in WSL2-Ubuntu-22.04-LTS.md](https://gist.github.com/ammarsufyan/51dd12d9471eb73b2348d373b605b45a)
- [[Linux] Ubuntu 22.04 NVIDIA 드라이버 + CUDA + cuDNN 설치하기](https://starlane.tistory.com/1)


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

---

tensorflow는 윈도우 GPU 지원을 2.10 이하 버전까지 지원한다. 윈도우에서 tensorflow를 사용하려면 wsl을 사용할 수 밖에 없다..
아래까지 윈도우에서 tenserflow를 지원하지 않는다라는 것을 모를 때까지 진행한 내용이다. 

CuDA, cuDNN, pytorch 설치
참고
- [Pytorch 설치 - CUDA Toolkit, cuDNN 설치](https://stat-thon.tistory.com/104) :  CUDA Toolkit, cuDNN 다운로드 방법
  ```
  nvcc --version # CUDA Toolkit 버전 확인
  ```
- [[ML][Windows 11] CUDA, cuDNN 설치](https://lonaru-burnout.tistory.com/16) : cuDNN 환경변수 설정 방법
- [torch.cuda.is_available()이 False일 때](https://neulvo.tistory.com/466) : CuDA, cuDNN에 맞는 pyTorch를 설치하는 방법 ([설치 명령어 생성기](https://pytorch.org/))
  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
주의사항
- GPU와 호환되는 CUDA Toolkit 버전으로 설치해야한다. ([호환 버전 확인](https://pytorch.org/get-started/locally/), [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive))
- CUDA : GPU에서 병렬 연산을 가능하게 해주는 플랫폼
- cuDNN : 딥러닝 연산을 최적화한 CUDA 라이브러리. 따라서 딥러닝을 GPU에서 실행하려면 CUDA와 함께 cuDNN도 반드시 설치해야 합니다.

UniGetUI로 다음의 패키지를 설치 : wget

문제 : 아래와 같이 명령어 출력이 깨짐. 
```
'head'��(��) ���� �Ǵ� �ܺ� ����, ������ �� �ִ� ���α׷�, �Ǵ� ��ġ ������ �ƴմϴ�.
```
해결방법 ([참고](https://blog.naver.com/PostView.naver?blogId=ycpiglet&logNo=223611366810)) : 제어판 > 모든 제어판 항목 > 국가 또는 지역 > 관리자 옵션 > 시스템 로캘 변경 > "Beta: 세계 언어 지원을 위해 Unicode UTF-8 사용" 체크

