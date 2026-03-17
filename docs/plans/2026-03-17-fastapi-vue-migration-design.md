# EVS Navigation — FastAPI + Vue 迁移设计

**日期**: 2026-03-17
**动机**: 性能（Streamlit rerun 模型）、UI/UX 限制、100+ 用户并发部署需求

---

## 决策：为什么不用 Java

所有核心 ML 处理为 Python-only（`torch`、`transformers`、`funasr`、`jieba`/`hanlp`）。
引入 Java 会形成三栈架构（Java + Vue + Python），维护成本翻三倍，且 Java 对 ML 层毫无加成。
FastAPI + Vue 保留 Python ML 代码，只引入一个新栈（Vue），是最优解。

---

## 整体分层架构

```
┌─────────────────────────────────────────────┐
│              Vue 3 SPA (前端)                │
│   Element Plus UI  │  Pinia 状态管理         │
│   Vue Router       │  Axios HTTP Client      │
└──────────────┬──────────────────────────────┘
               │ HTTPS REST + SSE
┌──────────────▼──────────────────────────────┐
│           FastAPI (API 层, stateless)        │
│   JWT Auth  │  File Upload  │  Task API     │
│   uvicorn × N workers (水平扩展)             │
└──────┬───────────────┬───────────────┬──────┘
       │               │               │
  PostgreSQL       Redis Broker    音频文件存储
  (业务数据)       (任务队列)      (本地/S3)
                       │
          ┌────────────▼──────────────────────┐
          │      Celery Workers (ML 层)        │
          │  worker-en:  CrisperWhisper (GPU)  │
          │  worker-zh:  FunASR               │
          │  worker-cpu: NLP / SI 分析         │
          └───────────────────────────────────┘
```

**核心原则**：FastAPI 层只做路由和协调，绝不在请求线程里运行 ML。
所有耗时操作全部交给 Celery worker 异步执行。

---

## 任务生命周期

```
前端                    FastAPI              Celery Worker
 │                        │                      │
 │── POST /tasks/asr ──→  │                      │
 │   (上传音频文件)        │── enqueue task ──→   │
 │← { task_id: "abc" } ── │                      │── 加载模型
 │                        │                      │── 转写中...
 │── GET /tasks/abc/stream → SSE { progress: 45% }│
 │                        │                      │── 完成
 │── GET /tasks/abc ───→  │←── save result ───── │
 │← { status: "done",     │                      │
 │    result_id: 123 } ── │                      │
 │── GET /results/123 ──→ │                      │
 │← { words: [...] } ──── │                      │
```

进度推送使用 **Server-Sent Events（SSE）**，单向推送，比 WebSocket 简单。

---

## 核心 API 端点

```
# 认证
POST   /auth/login              → { access_token }
POST   /auth/logout

# 任务
POST   /tasks/asr               → { task_id }
POST   /tasks/nlp               → { task_id }
GET    /tasks/{id}              → { status, progress, result_id }
GET    /tasks/{id}/stream       → SSE 进度流

# 结果
GET    /results/{id}            → 转写结果（分段 + 词级）
GET    /files                   → 文件列表
DELETE /files/{id}

# 分析
GET    /analysis/concordance
GET    /analysis/evs
POST   /analysis/si
```

---

## Vue 前端结构

```
src/
├── views/
│   ├── TranscribeView.vue    # ASR 上传 + 转写进度
│   ├── AnnotateView.vue      # EVS 标注（复用现有 Web Component）
│   ├── ConcordanceView.vue   # 语料分析
│   ├── NLPView.vue           # 中文 NLP 处理
│   ├── SIAnalysisView.vue    # SI 质量分析
│   ├── FilesView.vue         # 文件管理
│   └── LoginView.vue
├── components/
│   ├── TaskProgressBar.vue   # 复用：ASR/NLP 任务进度
│   ├── AudioUploader.vue     # 拖拽上传 + 预览
│   ├── ResultsTable.vue      # 转写结果展示
│   └── EvsAnnotator.vue      # 包装 evs_annotator Web Component
├── stores/                   # Pinia
│   ├── auth.js
│   ├── tasks.js              # 任务状态 + SSE 监听
│   └── files.js
└── api/
    └── client.js             # Axios + JWT 拦截器
```

**关键复用**：`evs_annotator/index.html` 是标准 Web Component，直接包装入 Vue：

```vue
<evs-annotator :data="JSON.stringify(pairs)" @change="onAnnotationChange" />
```

### 迁移优先级

| 优先级 | 页面 | 理由 |
|--------|------|------|
| 1 | Login + Files | 所有功能的入口 |
| 2 | TranscribeView | 核心功能，性能提升最明显 |
| 3 | AnnotateView | 可复用现有组件，成本低 |
| 4 | ConcordanceView | 纯数据展示，相对简单 |
| 5 | NLP / SI Analysis | 复杂度高，最后迁移 |

---

## 数据库迁移（SQLite → PostgreSQL）

SQLite 在 100+ 并发写入时会锁表，需替换。

**现有表**：`asr_files`、`asr_results_segments`、`asr_results_words`、`users`、
`login_history`、`si_analysis_results` — 全部保留，加索引优化。

**新增表**：
```sql
CREATE TABLE tasks (
    id          UUID PRIMARY KEY,
    type        VARCHAR(20),   -- 'asr', 'nlp', 'si'
    status      VARCHAR(20),   -- 'pending', 'running', 'done', 'failed'
    progress    INTEGER,
    user_id     INTEGER REFERENCES users(id),
    result_id   INTEGER,
    created_at  TIMESTAMP
);
```

Schema 版本管理使用 **Alembic**。

---

## 部署（Docker Compose）

```yaml
services:
  nginx:       # 反向代理，统一入口 :443
  frontend:    # Vue build → nginx 静态托管
  api:         # FastAPI × 2 实例
  worker-en:   # Celery，CrisperWhisper（需 GPU）
  worker-zh:   # Celery，FunASR
  worker-cpu:  # Celery，NLP/分析
  redis:       # 任务队列 + 结果缓存
  postgres:    # 数据库
```

**GPU 隔离**：`worker-en` 独占 GPU，避免显存抢占：
```yaml
worker-en:
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

---

## 工作量估算

| 阶段 | 内容 | 时间 |
|------|------|------|
| 1 | FastAPI 骨架 + Auth + 任务队列 | 2 周 |
| 2 | ASR/NLP Celery workers 迁移 | 1 周 |
| 3 | Vue 前端（5 个页面） | 3-4 周 |
| 4 | 数据迁移 + 集成测试 | 1 周 |
| **合计** | | **7-8 周** |
