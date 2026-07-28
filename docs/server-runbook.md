# GPU 서버 실행 런북

> 회사 GPU 서버(RTX A6000 ×2)에서 파이프라인을 돌리는 절차입니다.
> 이 문서는 실제로 겪은 함정과 해결책을 그대로 담았습니다.
>
> 이 서버는 개발·검증용입니다. 대회 본선 안심존은 완전히 별개(오프라인 폐쇄망·CPU·
> 사전신고 패키지)라서 GPU를 쓸 수 없고, 안심존에는 코드만 반입합니다.

---

## 0. 접속

```bash
ssh jhkim@222.103.107.7 -p 4132   # 방화벽 IP 제한 → 회사망에서만 접속 가능
nvidia-smi                        # A6000 ×2 보이면 정상
```

---

## 1. 환경 구축 (최초 1회)

### 먼저 알아둘 것 — 이 컨테이너의 제약

이 서버는 CUDA만 설치된 맨 컨테이너입니다. 처음 접속하면 이렇습니다:

| 상황 | 실제 |
|---|---|
| `python`, `pip` | 없음 (PATH에 CUDA 경로만 있음) |
| `git` | 없음 |
| `sudo apt-get install` | 작동하지 않음 — sudo는 되지만 `/var/cache/apt`에 쓸 권한이 없어 실패 |
| SSH 포트 포워딩 | 차단됨 (컨테이너에 sshd 없음, 게이트웨이 경유) |

→ 그래서 `/workspace`에 Miniconda를 직접 설치해서 씁니다. root 권한이 필요 없고,
`/workspace`는 영구 볼륨(3.5T)이라 유지됩니다.

### 1-1. Miniconda 설치

```bash
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda3
export PATH=/workspace/miniconda3/bin:$PATH

# 채널 이용약관 동의 (최신 conda는 이게 없으면 설치가 막힙니다)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 1-2. Python 3.11 환경 + git

기본 conda는 Python 3.14라 일부 패키지 wheel이 없을 수 있습니다. 3.11 환경을 따로 만듭니다.

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -y -n hf python=3.11
conda activate hf

conda install -y -c conda-forge git    # hf 환경 안에서 설치해야 합니다
git --version
```

> 자주 겪는 함정: git을 base 환경에만 깔면 `conda activate hf` 후에 `git: command not found`가
> 납니다. 반드시 `hf` 환경을 활성화한 상태에서 설치하세요.

### 1-3. 재접속해도 유지되게

```bash
/workspace/miniconda3/bin/conda init bash
echo 'conda activate hf' >> ~/.bashrc
```

### 1-4. 저장소 클론

```bash
cd /workspace
git clone https://github.com/BioCode67/hf-risk-prediction-sw.git
cd hf-risk-prediction-sw
git checkout claude/cardiac-arrest-early-warning-07fq9e

pip install -r requirements.txt
pip install lightgbm optuna catboost      # 노트북용 추가 패키지
python -c "import xgboost; print('xgboost', xgboost.__version__)"
```

---

## 2. VS Code 원격 접속 (권장)

터미널보다 훨씬 편합니다. 노트북을 바로 실행하고 그래프를 볼 수 있습니다.

### 2-1. SSH 설정

VS Code에서 `F1` → `Remote-SSH: Open SSH Configuration File` → `C:\Users\<사용자>\.ssh\config`

```
Host gpu-server
    HostName 222.103.107.7
    Port 4132
    User jhkim
```

> 함정: "Add New SSH Host"에 `ssh jhkim@222.103.107.7 -p 4132`를 통째로 붙여넣으면
> 호스트명이 `"222.103.107.7 -p 4132"`로 저장돼 접속이 실패합니다. 위처럼 config에 직접
> 작성하세요.

### 2-2. 접속

1. `F1` → `Remote-SSH: Connect to Host` → `gpu-server`
2. 플랫폼을 물으면 Linux (로컬 PC가 아니라 서버의 OS입니다)
3. `File > Open Folder` → `/workspace/hf-risk-prediction-sw`
4. Python·Jupyter 확장 설치 안내가 뜨면 설치

### 2-3. 접속이 "Waiting for port forwarding"에서 멈출 때

```
Ctrl+, (설정) → remote.SSH.useLocalServer  → 체크 해제
                remote.SSH.connectTimeout  → 120
```

그래도 안 되면 `F1` → `Remote-SSH: Kill VS Code Server on Host` 후 재접속,
또는 서버에서 `rm -rf ~/.vscode-server` 후 재접속.

### 2-4. 노트북 실행

탐색기에서 `notebooks/02_learning_project.ipynb` → 우측 상단 Select Kernel →
Python Environments → `hf` → Run All

---

## 3. Jupyter를 쓰고 싶다면 (VS Code가 안 될 때)

> 이 서버는 SSH 포트 포워딩이 막혀 있어 터널 방식이 작동하지 않습니다.
> `localhost:8888` 접속 시 `ERR_CONNECTION_RESET`이 뜹니다. 아래 방법을 쓰세요.

노트북을 헤드리스로 실행해 HTML로 뽑고, 로컬로 내려받아 브라우저로 봅니다.

```bash
# 서버에서
mkdir -p /workspace/out
jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=1800 \
  --allow-errors notebooks/02_learning_project.ipynb --output-dir /workspace/out
```

```powershell
# 로컬 PowerShell (scp는 포트 포워딩과 무관하게 작동합니다)
cd $HOME\Desktop
scp -P 4132 jhkim@222.103.107.7:/workspace/out/02_learning_project.html .
```

HTML을 더블클릭하면 모든 셀 결과와 그래프가 보입니다.

> 근본 해결을 원하면 서버 관리자에게 `AllowTcpForwarding yes` 허용 또는 8888 포트 매핑을
> 요청하세요. 그러면 VS Code Remote-SSH와 Jupyter 터널이 모두 정상 작동합니다.

---

## 4. 실행

### 4-1. 합성 데이터 스모크 테스트 (데이터 불필요)

GPU·Optuna 스택이 A6000에서 도는지 먼저 확인합니다.

```bash
python src/vitals_train.py --gpu --tune
# 다른 터미널에서: watch -n1 nvidia-smi   → 학습 중 GPU 점유 확인
```

- 정상이면 Optuna best CV AUPRC + XGBoost/NEWS 지표가 출력됩니다.
- `[gpu] CUDA unavailable ...`이 뜨면 CPU로 자동 폴백됩니다 → 드라이버/설치 점검.

그림까지 보려면:

```bash
python src/vitals_report.py      # PR·알람부담·궤적·lead-time 그림 → models/
python src/vitals_phenotype.py   # 표현형 히트맵
```

### 4-2. 공개 실데이터 — PhysioNet Challenge 2019 추천

인증 없이 받을 수 있고 양성과 대조군이 모두 있어서 오경보를 진짜로 측정할 수 있습니다.
패혈증 데이터지만 "활력징후 → 임박한 악화 사건" 구조가 같아 파이프라인이 그대로 돕니다.

```bash
cd /workspace
wget -r -N -c -np -nH --cut-dirs=4 \
  https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/
```

> 파일이 2만 개(각 5~10KB)라 전체 다운로드에 1시간 정도 걸립니다. 스모크 테스트는 수천 개로
> 충분하니, `nohup ... &`로 백그라운드에 걸어두고 먼저 실행해 보세요.

```bash
cd /workspace/hf-risk-prediction-sw

# 빠른 확인 (일부만)
python src/sepsis_explore.py /workspace/training_setA --max-files=1000 --horizon=6

# 본실행 (전체 + 튜닝 + GPU)
python src/sepsis_explore.py /workspace/training_setA --horizon=6 --tune --trials=50 --gpu
```

> `--horizon=6`을 반드시 붙이세요. 기본값 1시간은 환자당 양성 윈도우가 1개뿐이라
> 양성 비율이 0.2%가 되고 AUPRC가 기준선까지 붕괴합니다. 실측: 1h → AUPRC 0.003,
> 6h → 0.027 (기준선 0.012).

### 4-3. MIMIC-IV (인증 필요, 실제 심정지 표본)

```bash
python src/mimic_explore.py <경로>                 # 구조·활력징후 커버리지 확인
python src/mimic_explore.py <경로> --scan-arrest   # 심정지 itemid 후보 탐색
python src/mimic_explore.py <경로> --model --tune --gpu --trials=50   # 전체 실행
```

`icu/` 폴더가 있는 경로를 지정합니다. 한 번의 명령으로 튜닝 → 지표 → 알람부담 → lead-time →
그림 5종 → 표현형까지 나옵니다.

---

## 5. MIMIC-IV 접근 신청 (며칠~2주 소요)

1. PhysioNet 계정 생성 — https://physionet.org/register/
2. CITI 교육 이수 — "Data or Specimens Only Research" 과정 수료 후 완료 리포트(PDF) 확보
3. Credentialing 신청 — 프로필 작성 + 레퍼런스(지도교수/선배 연구원 등) 기재 후 심사 대기
4. 승인 후 DUA 서명 — https://physionet.org/content/mimiciv/
5. 다운로드

```bash
wget -r -N -c -np --user <PhysioNet-ID> --ask-password \
  https://physionet.org/files/mimiciv/3.1/
```

> 레퍼런스 확인 메일이 안 왔다면: 신청 즉시 발송되는 게 아니라 담당자가 검토를 시작할 때
> 발송되는 것으로 보입니다. 며칠 걸릴 수 있으니 스팸함을 확인하고, PhysioNet 계정의
> Credentialing 페이지에서 현재 단계를 직접 보세요.
>
> 대기하는 동안 §4-2(Challenge 2019)로 파이프라인과 수치를 먼저 확보합니다.

---

## 6. 산출물 회수

```bash
ls models/*.png models/*.json     # git-ignore 대상이라 커밋되지 않습니다
```

```powershell
# 로컬 PowerShell
scp -P 4132 jhkim@222.103.107.7:/workspace/hf-risk-prediction-sw/models/*.png ./
```

---

## 체크리스트

- [ ] `nvidia-smi`에서 A6000 ×2 확인
- [ ] Miniconda + `hf`(Python 3.11) 환경 + git 설치
- [ ] VS Code 원격 접속 (또는 nbconvert + scp 경로 확보)
- [ ] `--gpu --tune` 합성 스모크 테스트 통과 (GPU 점유 확인)
- [ ] Challenge 2019 `--horizon=6` 실행 → 실데이터 수치 확보
- [ ] (CITI 승인 후) 전체 MIMIC-IV `--model --tune --gpu`
- [ ] 결과 해석 → 제안서·발표 자료 반영

---

## 자주 겪는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `git: command not found` | base에만 설치됨 | `hf` 환경에서 `conda install -c conda-forge git` |
| `apt-get` 권한 오류 | 컨테이너에 apt 캐시 쓰기 권한 없음 | apt 대신 conda 사용 |
| VS Code "Waiting for port forwarding" | 포트 포워딩 차단 | `remote.SSH.useLocalServer` 해제, 또는 §3 방법 |
| Jupyter `ERR_CONNECTION_RESET` | SSH 터널 차단 | §3의 nbconvert + scp 방식 |
| SSH 끊기며 실행 중단 | 포그라운드 실행 | `tmux new -s run` 안에서 실행 |
| Optuna가 비정상적으로 느림 | `n_jobs=-1` 중첩 (코어 경합) | CV와 모델 모두 `n_jobs=1` |
| `git pull` 충돌 (untracked) | 서버에 같은 이름 파일 존재 | 백업 후 `mv` 하고 다시 pull |

> 안심존(본선)에서는 이 코드를 CPU·사전신고 패키지(numpy·pandas·scikit-learn·xgboost·shap·
> matplotlib)로 그대로 실행합니다. `--gpu` 옵션만 빼면 동일한 파이프라인입니다.
