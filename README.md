# 트랜스포머를 활용한 자연어 처리

이 저장소는 [트랜스포머를 활용한 자연어 처리](https://tensorflow.blog/transformer-nlp/) 책의 코드 저장소입니다.

<img alt="book-cover" height=400 src="https://tensorflowkorea.files.wordpress.com/2022/11/ed919ceca780_ed8ab8eb9e9cec8aa4ed8faceba8b8eba5bced999cec9aa9ed959cec9e90ec97b0ec96b4ecb298eba6ac.png" id="book-cover"/>

이 책의 코드는 주피터 노트북으로 제공하며 구글 코랩에서 테스트했습니다. 주피터 노트북마다 구글 코랩에서 실행할 수 있는 링크가 포함되어 있습니다.

로컬 컴퓨터에서 실행하려면 저장소를 클론하고 파이썬 가상 환경이나 콘다 환경을 만들어 실행하세요. 이 책의 코드는 라이브러리 버전에 따라 실행 결과가 달라질 수 있으므로 로컬에서 실행하는 경우 구글 코랩의 라이브러리 버전을 참고하세요.

# Setup - Ubuntu 24.04

## 요약

wsl2로 Ubuntu를 설치하여 CUDA 11.8버전과 cuDNN 8.7 버전을 설치했고, python 3.10 버전에서 poetry로 패키지 버전을 맞춰서 예제를 실행하는데 성공함.</br>
pyenv & poetry를 사용한 이유는 miniconda를 설치 후 예제에서 사용하는 패키지를 conda와 pip 명령어로 설치했는데 버전 관리가 안되서 챕터별 예제를 돌릴 때마다 버전 충돌이 남</br>

## 설치과정

### 1. wsl2 설치

관리자 권한의 터미널(PowerShell)에서 wsl2 설치 ([공식 설치 가이드](https://learn.microsoft.com/ko-kr/windows/wsl/install))

```sh
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

**설치방법**

1. [cnDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)에서 CUDA Toolkit 버전과 호환되는 것을 다운로드한다.
1. Ubuntu 홈으로 파일을 옮긴다.
1. 아래 명령어를 실행한다.

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
    sudo apt update
    sudo apt upgrade
    ```

**tensorflow와 torch가 의존하는 typing-extensions 버전 충돌 문제 해결**

- 문제 : 아래와 같이 요구하는 버전이 다름
  - pyTorch : typing-extensions==4.10.0
  - tensorflow : typing-extensions==4.5.0
- 현황
  - python==3.9.1
  - tensoflow==2.13
  - torch==2.6.0+cu118
  - cuda 버전 : 11.8
  - cuDNN 버전 : 8.6
- 결과 : tensoflow를 2.14.0, cuDNN을 8.7로 올리면  typing-extensions 버전 충돌이 발생하지 않을 것이다라는 가정으로 버전업을 했고 해결됨
  - [pyTorch==2.6.0](https://pypi.org/pypi/torch/2.6.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions>=4.10.0",
  - [tensoflow==2.13.0](https://pypi.org/pypi/tensorflow/2.13.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions (<4.6.0,>=3.6.6)",
  - [tensorflow==2.14.0](https://pypi.org/pypi/tensorflow/2.14.0/json)에서 요구하는 typing-extensions버전은 "typing-extensions (>=3.6.6)",
  - [tensorflow 버전 리스트](https://www.tensorflow.org/install/source?hl=ko#gpu_support_2)
        ![image](https://github.com/user-attachments/assets/854bf5a1-af04-419d-8982-b57652b3bf47)

**참고**

- [WSL2에 CUDA 설치하는 방법](https://webnautes.tistory.com/1848)
- [ammarsufyan/How to install CUDA-11.8 and CUDNN-8.6 for TensorFlow-2.13 in WSL2-Ubuntu-22.04-LTS.md](https://gist.github.com/ammarsufyan/51dd12d9471eb73b2348d373b605b45a)
- [[Linux] Ubuntu 22.04 NVIDIA 드라이버 + CUDA + cuDNN 설치하기](https://starlane.tistory.com/1)

### 5. pyenv 설치

pyenv는 python version manager이다. node.js에서 따지면 nvm이라고 할 수 있겠다.

**설치방법**

<https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv>

```sh
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv

$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
$ echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
$ echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc

$ source ~/.zshrc

$ pyenv --version
pyenv 2.5.4-1-gc579b636

# 빌드 도구 밒 라이브러리 설치
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

**배경**

- conda에 패키지가 별로 없어서 pip 패키지를 같이 사용하는데 패키지 의존성 관리가 전혀 안됨. 패키지 삭제시 관련된 패키지들이 삭제가 안됨.

**참고**

- [conda는 이제 그만 쓸래요 - pyenv & poetry](https://velog.io/@snoop2head/no-more-conda-please-pyenv-poetry-please)

### 6. pyenv-virtualenv 설치

<https://github.com/pyenv/pyenv-virtualenv>

```
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

source ~/.zshrc

pyenv virtualenvs
```

### 7. virtualenv 가상환경 생성 및 활성화

```sh
# 가상환경 생성 및 활성화
pyenv virtualenv 3.10 book-nlp-with-transormers
pyenv activate book-nlp-with-transormers
python --version
pip list

#upgrade pip
pip install --upgrade pip
```

특정 repository에 들어가면 virtual environment가 자동으로 실행되게 만들고 싶으면 다음과 같이 실행하면 된다.

```sh
# cd [REPOSITORY_PATH]
cd ~/Repository/book-nlp-with-transformers 
# pyenv local [ENVIRONMENT_NAME]
pyenv local book-nlp-with-transormers

# 특정 디렉토리(프로젝트)에서 사용하는 가상환경 확인
pyenv versions
```

### 8. poetry 설치 및 프로젝트 초기화

poetry는 각 프로젝트의 package version들을 명시하고, dependency들을 관리한다. node.js에서 따지면 npm이라고 할 수 있겠다.

```sh
pip install poetry
```

project repository로 가서 poetry 사용하도록 초기화한다.

```sh
poetry init
```

**참고**

- [poetry install 시 경고 메시지 안나오게 처리](https://blog.naver.com/drvoss/223523052028)
- [python - poetry 설치부터 project initializing, 활용하기](https://velog.io/@qlgks1/python-poetry-%EC%84%A4%EC%B9%98%EB%B6%80%ED%84%B0-project-initializing-%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0)
- <https://python-poetry.org/docs/pyproject/#dependencies-and-dependency-groups>
- [poetry 의 거의 모든것 (튜토리얼)](https://teddylee777.github.io/poetry/poetry-tutorial/)

### 9. python 3.10 설치

```sh
pyenv install --list | grep 3.10 # 설치할 버전 확인하려면 해당 명령어 실행
pyenv install 3.10
pyenv versions # 설치된 파이썬 버전 확인
```

**문제**

onnxruntime 1.20.1 버전 지원 문제 : poetry 사용시 onnxruntime 1.20.1이 python 3.9에서 설치안되서 3.10으로 올림

- <https://github.com/python-poetry/poetry/issues/10151>

### 10. 패키지 설치

**tensorflow 설치**

1. pyproject.toml 파일 수정
   - 파이썬 버전의 범위가 3.9 이상으로 잡혀있었는데, tensorflow 설치가 안됨

       ```bash
       requires-python = ">=3.10,<3.11"
       ```

   - tensorflow==2.14가 numpy 버전을 2 미만으로 지원해서 아래와 같이 추가함

       ```toml
       [tool.poetry.dependencies]
       tensorflow = "2.14"
       numpy = "^1.26"
       ```

2. 패키지 설치

    ```bash
    $ poetry install
    $ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ...
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] # 출력되면 정상
    ```

**torch 설치**

1. pyproject.toml 파일에 추가

    ```bash
    [[tool.poetry.source]]
    name = "pypi"
    priority = "primary"
    
    [[tool.poetry.source]]
    name = "pytorch"
    url = "https://download.pytorch.org/whl/cu118"
    priority = "supplemental"
    ```

2. 패키지 설치 ([설치 명령어](https://pytorch.org/))

    ```bash
    poetry add torch torchvision torchaudio --source pytorch
    python -c "import torch; print(torch.cuda.is_available())"
    ```

**torch-scatter 설치**

1. pyproject.toml 파일에 추가

    ```bash
    [[tool.poetry.source]]
    name = "torch-scatter"
    url = "https://data.pyg.org/whl/torch-2.1.0+cu118.html"
    priority = "supplemental"
    ```

2. 패키지 설치 ([설치 명령어](https://pypi.org/project/torch-scatter/))

    ```bash
    poetry add torch-scatter --source torch-scatter
    ```

**개발용 패키지 설치**

```bash
# Jupyter Notebook 관련 패키지 설치
poetry add -D notebook ipykernel ipywidgets
```

**실습용 패키지 설치**

```bash
# git-lfs(Large File Storage) 설치
poetry add git-lfs

# libsndfile 설치 (오디오 파일 처리) : libsndfile는 conda로 설치해야함. ChatGPT가 soundfile은 내부적으로 libsndfile을 사용한다고 해서 soundfile 설치
poetry add soundfile

# Base requirements
#poetry add "transformers[tf,torch,sentencepiece,vision,optuna,sklearn,onnxruntime]==4.11.3"
# tokenzer 설치안되서 버전 없앰. https://github.com/huggingface/tokenizers/issues/1087
poetry add "transformers[tf,torch,sentencepiece,vision,optuna,sklearn,onnxruntime]==4.11.3"
poetry add "datasets[audio]" matplotlib accelerate

# Chapter 1
#poetry add sacremoses

# Chapter 2 - Classification
poetry add "umap-learn==0.5.1" "wandb==0.18.7" 

# Chapter 3 - Anatomy
poetry add "bertviz==1.2.0"

# Chapter 4 - NER
poetry add "seqeval==1.2.2"

# Chapter 6 - Summarization
poetry add "nltk==3.6.6" "sacrebleu==1.5.1" "rouge-score==0.1.2" evaluate py7zr

# Chapter 7
poetry add "farm-haystack[elasticsearch7]=1.22.1" haystack

# Chapter 8
poetry add optuna onnxruntime onnx
 
# Chapter 9 - Few labels
poetry add "nlpaug==1.1.7" "scikit-multilearn==0.2.0" "faiss-cpu==1.7.4"

# Chapter 10 - Pretraining
poetry add psutil 
```

**문제해결**

*커널 충돌 문제*

예제를 실행시키다보면 커널충돌 에러가 발생함

```
현재 셀 또는 이전 셀에서 코드를 실행하는 동안 Kernel이 충돌했습니다. 
셀의 코드를 검토하여 가능한 오류 원인을 식별하세요. 
자세한 내용을 보려면 여기를 클릭하세요. 
자세한 내용은 Jupyter 로그를 참조하세요.
```

주피터노트북 최대 버퍼 늘리니 에러가 이전보다 자주 발생하지는 않음

```sh
# 주피터노트북 설정파일 생성
$ jupyter notebook --generate-config
Writing default config to: /home/yunheur/.jupyter/jupyter_notebook_config.py
# 2GB로 하니까 해결안되서 10GB로 설정하니까 안죽음
$ vim ~/.jupyter/jupyter_notebook_config.py
...
c.ServerApp.max_buffer_size = 10737418240 # 10GB
...
```

팁 : jupyter의 --paths 옵션을 사용하면 주피터 노트북이 참조하는 환경설정(config)파일들의 경로와 data파일의 경로들이 우선순위 순서로 출력됩니다. 각각 기능(config, data, runtime)별로 가장 위에 있는 경로에 들어가셔서 커스텀을 진행하면 됩니다.

```sh
!jupyter --paths
```

참고 : <https://bio-info.tistory.com/107>

---

*Ch2*

문제 : ValueError: Could not interpret optimizer identifier: <keras.src.optimizers.adam.Adam object at 0x7f5d87b7b490>

해결과정 :

1. ValueError: Could not interpret optimizer identifier: <keras.src.optimizers.adam.Adam object at 0x7f5d87b7b490>

    <https://discuss.huggingface.co/t/pretrain-model-not-accepting-optimizer/76209/26?page=2>

    transformers==4.50.0, accelerate==1.5.2 를 설치하니까 에러 발생함 (25.3.24기준 최신버전)

2. transformers 버전을 4.39.2으로 바꾸니까 다른 에러가 발생함.

    TypeError: Accelerator.**init**() got an unexpected keyword argument 'dispatch_batches'

    accelerate 버전을 0.30.0으로 변경

3. 다시 Adam 에러 발생함
4. transformers, accelerate 버전을 내릴 수 있을만큼 내려보자

    tensoflow 에서 요구하는 transformers 버전을 확인해보니 2.16미만까지 였음. 책에서는 4.11.3을 쓰고 있음.

    ```bash
    required by
     - tensorflow-text requires >=2.14.0,<2.15
     - transformers requires >=2.6,<2.16
    ```

    transformers 4.11.3은 tokenizers 0.10.3이 설치가 안됨

    transformers  4.33.1부터 되서 4.33.3 설치 accelerate는 0.20.3 이상이라서 0.20.3으로 설정함

결론 : transformers==4.33.3, accelerate==0.20.3으로 설정하니까 해결됨

---
*CH6*

![image](https://github.com/user-attachments/assets/19c854e6-a3c7-469d-a49e-636d88cd6ddf)

```bash
rouge-score = "0.0.4"
evaluate = "0.4.1"
```

버전을 올려니 해결됨

```bash
rouge-score = "^0.1.2"
evaluate = "^0.4.3"
```

---

*CH7*

1. farm-haystack을 설치하려고하니 wandb==0.19.8이 pydantic>=2.6,<3을 요구하기 때문에 에러가 발생함.

    [wandb==0.18.7](https://pypi.org/pypi/wandb/0.18.7/json)까지 제약사항이 없었으므로 wandb를 다운그레이드합니다.

2. poetry add fram-haystack를 실행하니 아래와 같은 에러가 발생함.

    ```bash
    Because no versions of farm-haystack match >1.26.4,<2.0.0
     and farm-haystack (1.26.4) depends on transformers (>=4.46,<5.0), farm-haystack (>=1.26.4,<2.0.0) requires transformers (>=4.46,<5.0).
    And because transformers[onnxruntime,optuna,sentencepiece,sklearn,tf,torch,vision] (4.33.3) depends on transformers (4.33.3), farm-haystack (>=1.26.4,<2.0.0) is incompatible with transformers[onnxruntime,optuna,sentencepiece,sklearn,tf,torch,vision] (4.33.3).
    So, because book-nlp-with-transformers depends on both transformers[onnxruntime,optuna,sentencepiece,sklearn,tf,torch,vision] (4.33.3) and farm-haystack (^1.26.4), version solving failed.
    ```

    farm-haystack 1.26.4 버전이 transformers의 버전 4.46 이상, 5.0 미만을 요구하는데, 프로젝트에서 사용 중인 transformers[onnxruntime,optuna,sentencepiece,sklearn,tf,torch,vision] (4.33.3)이 transformers 4.33.3 버전을 사용하고 있기 때문에 발생하는 문제입니다.

    transformer=4.33.3과 호환되는 [farm-haystack](https://pypi.org/pypi/farm-haystack/1.22.1/json) 버전이 없다…

    farm-haystack=1.22.1이 transformers=4.34.1을 지원하므로 4.33.3 → 4.34.1 로 업그레이드 함. datasets 버전이 충돌나서 2.14.7로 다운그레이드함.

    ```bash
    poetry add "farm-haystack=1.22.1"
    ```

    코드를 실행하니까 elasticsearch 클라이언트를 못찾아서 아래와 같이 수정함

    ```bash
    farm-haystack = {version = "1.22.1", extras = ["elasticsearch7"]}
    ```

3. docker로 실행한 es가 띄우면 죽음 (`TODO`)

    ![image](https://github.com/user-attachments/assets/4b91f651-5d39-47f2-9d86-5337215a07ea)

</br>

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
