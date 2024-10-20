
# 모델 학습을 위한 환경 설정

이 가이드는 AWS EC2 또는 호환되는 리눅스 환경에서 모델 학습을 위한 필수 환경을 설정하는 방법을 설명합니다. Python, Conda, CUDA, cuDNN 및 필요한 종속성을 포함합니다.

## 시스템 요구 사항

- **운영 체제**: Ubuntu 24.04
- **Python 버전**: 3.9.20
- **CUDA 버전**: 12.1.105
- **cuDNN 버전**: 8.9.7
- **Pip 버전**: 24.2
- **Conda 버전**: 24.7.1

## 1. 레포지토리 클론

먼저 레포지토리를 로컬 머신 또는 EC2 인스턴스에 클론합니다:

```bash
git clone https://github.com/your-repo.git
cd your-repo
```

## 2. Conda 설치 (설치되지 않은 경우)

Conda가 설치되어 있지 않다면, Miniconda를 설치합니다:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### Conda 버전 확인:

```bash
conda --version
```

Conda 24.7.1 이상의 버전이 필요합니다. 만약 Conda가 설치되어 있지 않거나 오래된 버전이라면, 위의 Miniconda 설치 방법을 따르거나 다음 명령어로 최신 버전으로 업데이트하세요:

```bash
conda update conda
```

## 3. Conda 환경 설정

`/docs/environment_setup` 디렉토리에 있는 `environment.yml` 파일을 사용해 Conda 환경을 재생성할 수 있습니다:

```bash
conda env create -f /docs/environment_setup/environment.yml
```

환경 생성 후, 아래 명령어로 활성화합니다:

```bash
conda activate <환경_이름>
```

## 4. 추가 Pip 종속성 설치

Conda 환경 외에, 일부 패키지는 `pip`를 통해 설치해야 할 수 있습니다. `/ML` 디렉토리에 있는 `requirements.txt` 파일을 사용해 추가 종속성을 설치합니다:

```bash
pip install -r /docs/environment_setup/pip_requirements.txt
```

## 5. CUDA 및 cuDNN 설치 (옵션)

AWS EC2 GPU 인스턴스를 사용하는 경우, CUDA 12.1과 cuDNN 8.9.7이 제대로 설치되어 있는지 확인해야 합니다:

### CUDA 설치 확인:

```bash
nvidia-smi
```

CUDA가 설치되어 있지 않다면, 다음 명령어로 설치할 수 있습니다:

```bash
conda install -c anaconda cudatoolkit=12.1
```

PyTorch가 CUDA를 잘 사용하는지 확인하려면 Python에서 아래 코드를 실행하세요:

```python
import torch
print(torch.cuda.is_available())  # True가 출력되면 GPU 사용 가능
```

## 6. Python, CUDA, cuDNN 버전

다음은 이 프로젝트에서 사용된 구체적인 버전입니다:

- **Python 버전**: 3.9.20
- **CUDA 버전**: 12.1.105
- **cuDNN 버전**: 8.9.7
- **Pip 버전**: 24.2
- **Conda 버전**: 24.7.1

코드가 원활하게 실행되려면 대상 머신에서 이 버전들이 일치해야 합니다.

## 7. Pip 수동 설치 (옵션)

Conda를 사용하지 않거나 추가적인 패키지가 필요한 경우, `/docs/environment_setup/pip_requirements.txt` 파일을 사용해 `pip`로 설치할 수 있습니다:

```bash
pip install -r /docs/environment_setup/pip_requirements.txt
```

이 명령어는 프로젝트에서 필요한 모든 Python 패키지를 설치합니다.

## 8. 설치 확인

마지막으로, 테스트 스크립트를 실행하거나 Python에서 GPU가 잘 인식되는지 확인하여 모든 것이 제대로 설치되었는지 확인합니다:

```python
import torch
print(torch.cuda.is_available())  # GPU가 사용 가능하면 True 출력
```

이 단계를 완료하면 AWS EC2 또는 호환되는 리눅스 머신에서 모델을 실행할 준비가 완료됩니다.
