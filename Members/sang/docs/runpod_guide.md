# SixSeven RunPod 사용 안내

팀 계정에서 필요한 GPU와 디스크를 직접 선택해 사용하면 됩니다. 결제와 계정 관리는 제가 담당하고, Pod 생성·접속·중지·삭제는 직접 진행하시면 됩니다.

## 1. 초대 수락 및 팀 계정 선택

1. 전달받은 **초대 링크**를 엽니다.
2. 초대받은 이메일로 RunPod에 가입하거나 로그인합니다.
3. **Join Team**을 눌러 초대를 수락합니다.
4. 오른쪽 위 프로필 메뉴에서 **SixSeven 팀 계정**을 선택합니다.

Pod를 만들기 전에 **개인 계정이 아닌 SixSeven으로 선택되어 있는지** 확인해주세요. Developer 권한으로 GPU·디스크 선택과 Pod 생성·시작·중지·삭제가 가능합니다.

초대 링크가 만료되거나 팀이 보이지 않으면 저에게 알려주세요. [공식 팀 계정 안내](https://docs.runpod.io/accounts-billing/manage-accounts)

## 2. Pod 생성

Pod는 코드를 실행할 GPU 서버입니다.

1. [RunPod 콘솔](https://console.runpod.io/)에서 **Pods → Deploy**로 이동합니다.
2. **Workload**에서 작업에 맞는 템플릿을 선택합니다.
   - PyTorch 작업이라면 공식 PyTorch 템플릿으로 시작하면 됩니다.
   - Python·PyTorch·CUDA 버전이 사용할 코드와 맞는지 확인해주세요.
   - `Deprecated` 표시가 있는 템플릿은 새 작업에 사용하지 않는 편이 좋습니다.
3. **Pod name**을 알아보기 쉽게 지정합니다.
   - 예: `이름-프로젝트명`
4. **Compute**에서 GPU 종류와 GPU 개수를 선택합니다.
   - 모델에 필요한 VRAM과 실행 비용을 함께 확인합니다.
   - GPU 개수를 늘려도 코드에 따라 자동으로 여러 GPU를 사용하지 않을 수 있습니다.
5. **Storage**에서 저장 공간을 설정합니다.
6. 오른쪽 **Total / hour**와 디스크 구성을 확인한 뒤 **Deploy Pod**를 누릅니다.
7. Pods 화면에서 초기화가 끝날 때까지 기다립니다.

생성이 진행되지 않으면 Pod의 **Logs**에서 이미지 다운로드나 실행 오류를 확인할 수 있습니다. [공식 Pod 관리 안내](https://docs.runpod.io/pods/manage-pods)

## 3. 디스크 선택 및 파일 저장

| 종류 | 용도 | 데이터 유지 |
|---|---|---|
| **Container disk** | 운영체제, 임시 파일, 패키지 설치 공간 | 중지 시 데이터가 사라질 수 있음 |
| **Volume disk** | 모델·데이터·체크포인트 보관 | 중지 후 유지, **Pod 삭제 시 함께 삭제** |
| **Network volume** | Pod를 바꿔도 유지할 별도 저장 공간 | Pod 삭제 후에도 유지, 별도 비용 발생 |

**`/workspace`에 실제 Volume disk 또는 Network volume이 연결되어 있는지 먼저 확인해주세요.** 폴더 이름이 `/workspace`라고 해서 자동으로 영구 저장되는 것은 아닙니다.

저장소가 연결되어 있다면 코드·데이터·모델·학습 결과는 `/workspace` 아래에 저장해주세요. 중요한 결과는 로컬이나 별도 저장소에도 백업해주세요.

Network volume을 연결하면 일반 Volume disk 대신 `/workspace`에 마운트됩니다. 기존 Pod와 독립적으로 유지되지만, 사용할 데이터센터와 GPU 재고도 함께 확인해야 합니다. [공식 저장소 안내](https://docs.runpod.io/pods/storage/types)

## 4. 접속 및 작업

**브라우저로 사용할 때**

1. **Pods → 사용할 Pod → Connect**를 엽니다.
2. **HTTP Services → Jupyter Lab** 링크를 누릅니다.
3. 인증을 요구하면 템플릿 안내나 Pod 로그에 표시된 비밀번호·토큰을 확인합니다.
4. JupyterLab에서 노트북, 파일 업로드, 터미널을 사용하면 됩니다.

처음에는 JupyterLab이 포함된 공식 PyTorch 템플릿을 사용하면 접속이 간단합니다.

**VS Code나 SSH로 사용할 때**

본인 컴퓨터에서 SSH 키를 준비하고 공개키를 RunPod에 등록한 뒤, **Connect** 화면의 SSH 안내에 따라 접속합니다. VS Code에서는 **Remote - SSH** 확장을 사용하면 됩니다. SSH 개인키는 다른 사람에게 전달하지 마세요.

장시간 실행할 학습은 SSH에서 `tmux` 같은 도구를 사용해 세션을 관리하고, 로그와 체크포인트를 남겨주세요. [공식 접속 안내](https://docs.runpod.io/pods/connect-to-a-pod)

## 5. 작업 종료와 재접속

### 나중에 같은 환경을 다시 사용할 때: Stop

1. 필요한 파일과 결과를 저장합니다.
2. 진행 중인 학습을 중단해도 되는지 확인합니다.
3. **Pods → 해당 Pod → Stop(■)**을 누릅니다.
4. 실행이 멈춘 상태인지 확인합니다.

**브라우저나 VS Code를 닫는 것만으로는 Pod가 중지되지 않습니다.** 실행 중이면 비용이 계속 발생합니다. Stop 후에도 Volume disk와 별도 저장소 비용은 발생합니다.

다시 사용할 때는 **Start(▶)**를 누르고 준비가 끝나면 **Connect**로 접속합니다. 중지 중 GPU가 다른 사용자에게 할당되면 같은 Pod를 바로 시작하지 못할 수도 있습니다. 이 경우 데이터를 백업한 뒤 다른 Pod로 옮기는 방법을 확인해주세요.

### 더 이상 사용하지 않을 때: Terminate

1. 코드·데이터·모델·결과물을 외부에 백업합니다.
2. 백업 파일을 실제로 열거나 확인합니다.
3. 해당 Pod의 **Terminate(휴지통)**를 누릅니다.

**Terminate하면 일반 Volume disk의 `/workspace` 데이터도 삭제되며 복구할 수 없습니다.** Network volume은 별도로 남아 비용이 발생하므로, 필요 없을 때 따로 정리해야 합니다. [공식 중지·삭제 안내](https://docs.runpod.io/pods/manage-pods)

팀 계정에는 다른 사람의 리소스도 보일 수 있으니 **본인 Pod 이름을 확인한 뒤 조작**해주세요. 접속이나 생성이 막히면 Pod 이름과 오류 화면을 보내주시면 같이 확인하겠습니다.
