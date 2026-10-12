# Duplex Arena

**A 3D acoustic playground for full-duplex speech agents.**

여러 음성 모델이 같은 가상 공간에서 동시에 듣고 말하는 연구 환경입니다.
캐릭터의 발화와 수신 상태, 공간 내 소리 전달, 겹침, 개별 마이크 음량을
하나의 화면에서 관찰하고, 같은 음향 엔진을 headless 평가와 RL rollout에 사용합니다.

## 바로 실행

Python 3.11+ (모델 서버는 3.12 권장), Node.js 22+가 필요합니다.

```bash
git clone https://github.com/KyuDan1/duplex-arena.git
cd duplex-arena
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cd web
npm ci
npm run build
cd ..
duplex-arena serve
```

브라우저에서 **http://127.0.0.1:8765** → **Run**.
GPU나 모델 다운로드 없이 실행됩니다. 개발 중에는 별도 터미널에서
`cd web && npm run dev`를 실행하면 5173 포트의 뷰어가 8765 API에 연결됩니다.

## 구현된 기능

- Three.js 3D 공간, 모델별 로봇 캐릭터, 발화·수신 표시, 소리 전파 효과, 시점 회전과 확대.
- 개별 에이전트 위치·음소거, 거리 감쇠, 샘플 단위 전파 지연, 칸막이 차폐, 한 번의 벽 반사, 독립적인 마이크 잡음.
- Roundtable / Interruption / Cocktail party / Quiet baseline 장면, 실행·일시정지·프레임 진행·시드 초기화.
- 에이전트별 송신/수신 dBFS, 중첩 비율, 보상 항목, 타임라인, 실제 수신 PCM 모니터와 WAV/JSON 내보내기.
- Moshi / PersonaPlex의 native Opus WebSocket 연결, 사용자 모델용 PCM WebSocket 규약, CPU echo 워커.
- 선택적 실제 수신 ASR 관찰기: [ASR 설정](docs/ASR.md).
- 24 kHz / 80 ms 고정 시간 간격의 다중 에이전트 환경과 실행 가능한 NumPy on-policy 학습 예제.

**현재 데모는 합성 음색과 미리 쓴 대사입니다.** PersonaPlex/Moshi 가중치를 실행한 결과가
아니며, 대사는 음성 인식 결과가 아닙니다. 실제 모델 모드는 별도의 워커를 연결해야 합니다.
네이티브 모델이 내보내는 텍스트는 생성 발화 텍스트이고, 수신 ASR과 별개입니다.

## GPU 서버로 옮기기

시뮬레이터와 웹 화면은 CPU에서 실행하고, GPU마다 모델 서버를 별도 프로세스로 띄웁니다.
모델별 Python 환경도 분리합니다. PersonaPlex와 Moshi는 같은 `moshi` 패키지 이름을
사용하므로 한 환경에 동시에 설치하지 않습니다.

```bash
pip install -e '.[models]'
# 각 모델 서버를 먼저 시작하고 configs/workers.example.json의 주소와 프롬프트를 수정
duplex-arena serve --models configs/workers.example.json
```

정확한 실행 명령과 CUDA 배치는 [GPU 서버 가이드](docs/GPU_SERVER.md),
모델별 프로토콜과 제약은 [모델 연결 문서](docs/MODELS.md)에 있습니다.
모델 하나가 GPU 하나에 들어가는지는 VRAM, 정밀도, 컨텍스트 길이에 따라 달라집니다.
실제 B200에서 PersonaPlex–Moshi의 60초 대화와 수신 WAV/ASR을 검증했습니다.
측정 결과와 Slurm 재실행 방법은 [B200 검증 기록](docs/B200_VERIFICATION.md)에 있습니다.

CPU만으로 실제 네트워크 경로를 먼저 확인할 수 있습니다.

```bash
# 터미널 1: 연결·오디오 왕복용 워커 (언어 모델 아님)
python -m workers.echo_worker --port 9001
# 터미널 2
duplex-arena serve --models configs/echo.example.json
```

## 평가와 on-policy 학습

```bash
# 재현 가능한 데모 평가. 각 마이크 WAV + 전체 trace.jsonl + summary.json
duplex-arena evaluate --scenario cocktail --seed 42 --seconds 18 --output runs/cocktail

# 실제 워커 평가: 송신/수신/모델 입력 WAV와 전체 trace를 wall-clock 1×로 기록
duplex-arena evaluate --models configs/local-workers.json --seconds 60 --output runs/live

# 종료된 GPU 작업의 실제 오디오·3D 상태를 CPU에서 재생
duplex-arena serve --replay runs/live --port 18766

# 작은 speak/listen gate 정책의 실제 REINFORCE 업데이트 (CPU)
python examples/on_policy.py --episodes 12 --output runs/gates

# 수치·프로토콜·서버 검증
pytest -q
```

```python
import numpy as np
from duplex_arena.env import DuplexArenaEnv

env = DuplexArenaEnv(seed=42, horizon=750)
observations, infos = env.reset()
while env.agents:
    # 실제 정책은 observations[id]["audio"]를 받아 다음 PCM 프레임을 만듭니다.
    actions = {agent: np.zeros(env.frame_size, dtype=np.float32) for agent in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
env.close()
```

학습 예제는 SLM 전체를 학습하지 않습니다. 정책 버전을 rollout 동안 고정하고,
행동 확률·개별 보상·discounted return을 이용해 작은 발화 gate를 학습합니다.
실제 SLM의 on-policy 학습에는 토큰 행동, behavior log-probability,
value estimate, policy-version synchronization을 모델 학습 코드에서 추가해야 합니다.
환경은 PettingZoo와 유사한 dict 인터페이스이며 정식 등록 환경은 아닙니다.

## 해석 범위

- 소리 효과는 보기 좋게 느리게 표현합니다. 실제 전파 지연은 **343 m/s**로 계산합니다.
- 음량은 **dBFS**입니다. 실제 SPL/dBA 측정치가 아닙니다.
- `listening`은 수신 오디오의 에너지 활동 표시입니다. 모델이 이해했다는 증거가 아닙니다.
- 반사는 오른쪽 벽의 한 image source이며, 정밀 RIR/회절/HRTF/고차 잔향은 구현하지 않았습니다.
- self sidetone과 self echo는 제외합니다. 타인 음성의 반사음은 포함합니다.
- 보상은 audibility·overlap·energy의 공개된 음향 휴리스틱입니다. 대화의 의미나 자연스러움을 평가하지 않으며, 유효한 맞장구도 overlap cost를 받을 수 있습니다.
- 브라우저 세션은 하나의 공유 실험을 관찰합니다. 기본 episode는 60초, UI WAV 보관은 최근 60초입니다. 긴 평가는 headless 모드에서 파일로 기록합니다.
- 실제 모델 연결은 wall-clock 1×입니다. 수신·송신을 독립 처리하고 underrun/overflow를 기록합니다. 정확한 학습 rollout은 직접 환경 인터페이스를 사용하는 별도 trainer가 담당합니다.

[구조와 개발 계획](docs/ARCHITECTURE.md) · [GPU 서버](docs/GPU_SERVER.md) · [모델 연결](docs/MODELS.md) · [ASR](docs/ASR.md)

## 참고한 원본

- [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)

프로젝트 이름과 캐릭터 라벨은 연결 대상 식별용이며 각 모델 제공자와의 제휴를 의미하지 않습니다.
