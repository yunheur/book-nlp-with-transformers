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





https://gist.github.com/ammarsufyan/51dd12d9471eb73b2348d373b605b45a
https://webnautes.tistory.com/1873


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

