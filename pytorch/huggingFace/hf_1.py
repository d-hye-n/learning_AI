import torch

# 1. CUDA 사용 가능 여부 확인 (True가 나와야 함)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 2. 사용 가능하다면, 설치된 CUDA 버전 확인
if cuda_available:
    # PyTorch 빌드에 사용된 CUDA 버전
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")

    # 현재 시스템의 NVIDIA 드라이버가 지원하는 CUDA 버전
    # (nvidia-smi 명령어로 확인하는 것과 동일한 정보)
    # 이 기능은 최신 PyTorch 버전에만 있을 수 있습니다.
    try:
        import subprocess

        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'],
                                capture_output=True, text=True)
        driver_version, cuda_driver_version = result.stdout.strip().split(', ')
        print(f"CUDA Driver Version: {cuda_driver_version}")
    except (FileNotFoundError, ValueError):
        print("Could not run 'nvidia-smi' to check the driver's CUDA version. Please check manually.")

    # 3. 사용 중인 GPU 장치 정보 확인
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")