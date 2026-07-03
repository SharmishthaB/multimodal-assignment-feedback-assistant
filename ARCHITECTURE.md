# MAFA Full-Stack Architecture

## 🏗️ Project Structure

```
mafa/                               # Root monorepo
├── frontend/                       # React + TypeScript (Vite)
│   ├── src/
│   │   ├── pages/                 # Route pages (Dashboard, Assignment, etc)
│   │   ├── components/            # Reusable React components
│   │   ├── hooks/                 # Custom hooks
│   │   ├── types/                 # TypeScript types
│   │   └── utils/                 # Utilities
│   ├── vite.config.ts
│   └── package.json
│
├── backend/                        # Express API + TypeScript
│   ├── src/
│   │   ├── index.ts              # Main entry
│   │   ├── routes/               # API routes
│   │   ├── controllers/          # Business logic
│   │   ├── middleware/           # Auth, validation
│   │   ├── types/                # TS interfaces
│   │   └── utils/                # Helpers
│   ├── tsconfig.json
│   └── package.json
│
├── ml-service/                     # Python ML pipeline
│   ├── src/
│   │   ├── pipeline.py           # LangGraph orchestration
│   │   ├── vision/               # Vision models
│   │   ├── llm/                  # LLM components
│   │   └── rag/                  # RAG system
│   ├── requirements.txt
│   └── Dockerfile
│
├── packages/
│   └── shared/                     # Shared types & utilities
│       ├── types.ts              # Shared TypeScript types
│       └── package.json
│
├── scripts/                        # Build & deploy scripts
├── docker-compose.yml             # Full-stack orchestration
├── package.json                   # Monorepo configuration
├── tsconfig.json                  # Root TypeScript config
├── .env.example
└── README.md
```

---

## 🔄 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React/Vite)                    │
│  - Dashboard (upload, manage assignments)                  │
│  - Assignment detail (view results, feedback)              │
│  - Auth flows (Supabase Auth)                              │
└────────────────────┬──────────────────────────────────────┘
                     │ Axios + API client
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  BACKEND (Express/Node)                     │
│  - REST API (/api/assignments, /api/feedback)              │
│  - Request validation & auth                               │
│  - File upload handling                                    │
└────────────────┬────────────────────────────────────────────┘
                 │ REST calls
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              ML-SERVICE (Python/FastAPI)                    │
│  - Vision transcription (Qwen2-VL)                         │
│  - Rubric retrieval (ChromaDB)                             │
│  - Assignment analysis (Phi-3 LLM)                         │
│  - Socratic coaching                                       │
└────────────────┬────────────────────────────────────────────┘
                 │ 
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              DATABASE (Supabase/PostgreSQL)                 │
│  - Users (auth)                                            │
│  - Assignments (metadata, submissions)                    │
│  - Feedback (analysis results)                            │
│  - Rubrics (grading criteria)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + TypeScript + Vite + React Router |
| **Backend** | Express + TypeScript + Zod (validation) |
| **Database** | Supabase (PostgreSQL + Auth) |
| **ML Service** | Python + LangGraph + Transformers |
| **Auth** | Supabase Auth (JWT) |
| **Deploy** | Docker Compose + Render/Railway |
| **CI/CD** | GitHub Actions |

---

## 📦 Milestones Progress

- **M1** ✅ Project skeleton (monorepo, config, base files)
- **M2** ⏳ Frontend baseline (auth UI, dashboard, components)
- **M3** ⏳ Backend API + DB (Supabase setup, CRUD routes)
- **M4** ⏳ Tests & CI (Jest, GitHub Actions)
- **M5** ⏳ Dockerize & deploy
- **M6** ⏳ Polish & resume assets

---

## 🔗 Key Files

### Frontend
- `frontend/src/App.tsx` — Main router
- `frontend/src/pages/Dashboard.tsx` — Assignment management
- `frontend/src/pages/Assignment.tsx` — Results & feedback

### Backend
- `backend/src/index.ts` — Express setup + routes
- `backend/src/routes/assignments.ts` — Assignment CRUD
- `backend/src/middleware/auth.ts` — Auth middleware

### ML Service
- `ml-service/src/pipeline.py` — LangGraph pipeline
- `ml-service/src/api.py` — FastAPI server (M3)

### Config
- `.env.example` — Environment variables template
- `docker-compose.yml` — Multi-service orchestration

---

## 💻 Quick Start

### Install dependencies
```bash
npm install  # Installs all workspaces
```

### Development
```bash
npm run dev              # Runs frontend + backend in parallel
npm run dev:frontend    # Frontend only (port 5173)
npm run dev:backend     # Backend only (port 3000)
```

### Build
```bash
npm run build           # Build frontend + backend
npm run build:frontend
npm run build:backend
```

### Docker
```bash
docker-compose up --build
```

---

## 📝 Next Steps (M2)

1. Create React pages: Dashboard, Login, AssignmentDetail
2. Set up Supabase Auth integration
3. Create API client (Axios instance)
4. Build responsive UI with Tailwind CSS
5. Implement file upload component

---

**Status**: M1 Complete ✅ → Ready for M2 Frontend Development
