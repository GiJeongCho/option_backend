# NETBITEX 옵션 외부 배포·Cloudflare Tunnel 연결 가이드

> 작성일: 2026-10-23  
> 목표 화면: `https://netbitex.com/option/`  
> 전략 화면: `https://netbitex.com/option/#strategy-analysis`  
> API 주소: `https://api.netbitex.com`

## 1. 최종 구성

- 기존 `netbitex.com`: GitHub Pages 유지
- `netbitex.com/option/`: Vite 프론트엔드 빌드 결과 배포
- `api.netbitex.com`: Cloudflare Tunnel을 통해 이 PC의 FastAPI 연결
- 로컬 FastAPI: `http://127.0.0.1:8713`
- 원본 옵션·현물·선물 자료: 현재 PC의 `backend/rowdata`

GitHub Pages는 화면만 제공한다. 데이터 조회와 백테스트 API는
Cloudflare Tunnel이 현재 PC로 전달한다. 공유기 포트포워딩이나
공인 고정 IP는 필요하지 않다.

## 2. 현재 확인된 상태

2026-10-23 기준으로 다음 상태를 직접 확인했다.

- Windows 서비스 `cloudflared`: `Running`
- 시작 유형: `Automatic`
- 로컬 API `http://127.0.0.1:8713/api/health`: 정상
- 옵션 원자료 파일: 769개
- 전략 재생성 상태: 최신 자료까지 완료

단, FastAPI 자체는 현재 터미널에서 실행 중이므로 Windows 재부팅
후에는 별도 자동 실행 설정이 필요하다.

## 3. 가장 먼저 해야 할 보안 조치

Cloudflare가 보여 주는 다음 형식의 값은 비밀 토큰이다.

```text
cloudflared.exe service install <TUNNEL_TOKEN>
```

`<TUNNEL_TOKEN>`은 비밀번호와 동일하게 취급한다.

- Markdown, 소스코드, Git, 메신저, 채팅에 넣지 않는다.
- 전체 설치 명령이 보이는 화면을 캡처하지 않는다.
- 노출된 토큰을 다시 사용하지 않는다.

현재 선택한 터널 이름은 `option2`이며 터널 자체는 그대로 사용한다.
다만 설치 명령의 `eyJ...` 문자열은 일부 키가 아니다. Cloudflare
공식 문서상 그 값 하나만으로 connector를 실행할 수 있는 완전한
터널 토큰이다. 따라서 터널은 유지하고 토큰만 다음처럼 회전한다.

1. Cloudflare → `Networking` → `Tunnels`
2. `option2` 선택
3. `Overview` → `Refresh token` 또는 `Rotate token`
4. 새 설치 명령은 관리자 PowerShell에서만 실행
5. 새 토큰으로 로컬 connector를 다시 설치
6. 기존 connector 연결이 있다면 종료한 뒤 새 연결이 `Healthy`인지
   확인

실제 새 토큰은 이 문서나 채팅에 기록하지 않는다.

기존 `option` 터널 토큰도 과거 문서에 저장된 적이 있다. 해당
터널을 계속 사용할 경우 반드시 토큰을 회전해야 하며, 필요 없다면
새 터널이 정상 동작한 뒤 삭제한다.

## 4. Windows connector를 새 터널로 교체

Windows에는 이미 `cloudflared` 서비스가 하나 설치되어 있다.
따라서 새 터널 설치 명령을 바로 실행하지 말고 기존 서비스를 먼저
제거한다.

관리자 PowerShell:

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" service uninstall
```

그다음 Cloudflare의 `option2` 터널 화면에 표시된 Windows
설치 명령을 관리자 PowerShell에서 실행한다.

```text
cloudflared.exe service install <새 비밀 토큰>
```

문서에는 실제 토큰을 적지 않는다.

설치 확인:

```powershell
Get-Service cloudflared
```

정상 결과:

- `Status`: `Running`
- `StartType`: `Automatic`
- Cloudflare 터널 상태: `Healthy`

`option2`가 `Healthy`가 되기 전에는 기존 터널을 삭제하지 않는다.

## 5. `api.netbitex.com` 경로 추가

새 터널 상세 화면에서 다음 순서로 설정한다.

1. `Routes` → `Add route`
2. `Published application` 선택
3. Hostname:
   - Subdomain: `api`
   - Domain: `netbitex.com`
   - Path: 비워 둠
4. Service:
   - Type: `HTTP`
   - URL: `http://localhost:8713`
5. 저장

최종 연결은 다음과 같다.

```text
https://api.netbitex.com
        ↓ Cloudflare HTTPS/Tunnel
http://127.0.0.1:8713
```

같은 호스트 이름의 기존 DNS 레코드가 있다는 오류가 나오면
Cloudflare DNS에서 기존 `api` 레코드를 확인한다. 다른 서비스에서
사용 중이 아니라면 기존 잘못된 레코드를 제거한 뒤 Route를 다시
저장한다.

## 6. FastAPI 실행

백엔드 터미널:

```powershell
cd C:\project\option\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8713
```

운영 중에는 `--reload`를 사용하지 않는다. `--reload`는 개발 편의를
위한 파일 감시 기능이며 프로세스를 재시작할 수 있다.

로컬 확인:

```text
http://127.0.0.1:8713/api/health
```

외부 확인:

```text
https://api.netbitex.com/api/health
https://api.netbitex.com/docs
```

외부 `/api/health`에서 `"status": "ok"`가 나오면 터널 연결이
완료된 것이다.

## 7. 프론트엔드 운영 설정

현재 개발 환경은 Vite가 `/api`를 로컬 `8713` 포트로 프록시한다.
GitHub Pages에서는 Vite 개발 프록시가 없으므로 다음 코드 변경이
필요하다.

1. Vite 공개 기본 경로를 `/option/`으로 설정
2. 운영 API 기본 주소를 `https://api.netbitex.com`으로 설정
3. 모든 `fetch('/api/...')` 호출에 운영 API 주소를 적용
4. FastAPI CORS에 다음 Origin 허용
   - `https://netbitex.com`
   - `https://www.netbitex.com`
5. 로컬 개발 주소는 계속 허용
   - `http://127.0.0.1:5173`
   - `http://localhost:5173`

이 변경은 실제 API Route가 외부에서 정상 동작하는 것을 확인한
뒤 코드에 적용한다.

## 8. 프론트엔드 빌드와 GitHub Pages 배포

코드 변경 뒤:

```powershell
cd C:\project\option\frontend
npm run build
```

생성 결과:

```text
C:\project\option\frontend\dist
```

`dist` 안의 내용을 기존 `netbitex.com` 홈페이지 저장소의
`option` 폴더에 복사한다.

```text
홈페이지저장소/
├─ 기존 홈페이지 파일
└─ option/
   ├─ index.html
   └─ assets/
```

홈페이지 저장소에서:

```powershell
git add .
git commit -m "Add option analysis app"
git push
```

배포 확인:

```text
https://netbitex.com/option/
https://netbitex.com/option/#strategy-analysis
```

## 9. Windows 재부팅 뒤 자동 복구

Cloudflare Tunnel은 Windows 서비스 `Automatic`으로 설치되므로
자동 실행된다. FastAPI는 별도로 작업 스케줄러에 등록해야 한다.

작업 스케줄러 권장 설정:

- 트리거: 시스템 시작 시
- 사용자 로그인 여부와 관계없이 실행
- 시작 위치: `C:\project\option\backend`
- 프로그램: 현재 Conda/Python 환경의 `python.exe`
- 인수:

```text
-m uvicorn app.main:app --host 127.0.0.1 --port 8713
```

Python 실행 파일 위치는 다음 명령으로 확인한다.

```powershell
where.exe python
```

## 10. 장애 확인 순서

### `502 Bad Gateway`

Cloudflare Tunnel은 살아 있지만 FastAPI가 꺼졌을 가능성이 크다.

```powershell
Invoke-RestMethod http://127.0.0.1:8713/api/health
```

실패하면 FastAPI를 다시 실행한다.

### `api.netbitex.com`을 찾을 수 없음

- Cloudflare 도메인 상태가 `Active`인지 확인
- Tunnel Route의 hostname 확인
- Cloudflare DNS에 `api` 충돌 레코드가 있는지 확인

### 브라우저 CORS 오류

- FastAPI의 허용 Origin에 `https://netbitex.com`이 있는지 확인
- 프론트 API 주소가 `https://api.netbitex.com`인지 확인
- 백엔드 재시작

### 옵션 화면은 뜨지만 `/api`가 404

GitHub Pages에는 `/api` 프록시가 없다. 프론트가 상대주소
`/api/...`가 아닌 `https://api.netbitex.com/api/...`를 사용하도록
운영 빌드 설정을 확인한다.

## 11. 보안 주의사항

- CORS는 인증 기능이 아니다.
- API를 공개하면 누구나 직접 API URL을 호출할 수 있다.
- Cloudflare Rate Limiting 또는 애플리케이션 인증을 추가한다.
- 불필요하면 운영 환경에서 `/docs`를 비활성화한다.
- `rowdata` 파일 자체를 정적 웹 폴더에 복사하지 않는다.
- Cloudflare Tunnel 사용 시 공유기 8713 포트를 외부에 개방하지
  않는다.
- 토큰이 노출되면 즉시 터널 삭제 또는 토큰 회전 후 connector를
  재설치한다.

## 12. 완료 체크리스트

- [ ] `option2` 터널 토큰 회전
- [ ] 과거 토큰이 노출된 기존 `option` 터널 회전 또는 폐기
- [ ] 새 안전한 터널 connector `Healthy`
- [ ] `api.netbitex.com → http://localhost:8713` Route 저장
- [ ] 외부 `/api/health` 정상
- [ ] 프론트 API 주소·Vite `/option/`·CORS 수정
- [ ] 프론트 빌드 성공
- [ ] 홈페이지 저장소 `option` 폴더 배포
- [ ] 외부 옵션 화면과 전략 분석 확인
- [ ] FastAPI Windows 자동 실행 등록
