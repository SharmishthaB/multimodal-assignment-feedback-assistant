# MAFA - Full Stack Web App

Assignment grading system with React frontend, Node backend, and ML pipeline for feedback generation.

## Overview

MAFA combines a modern web interface with AI-powered assignment grading. Students upload assignments, the system extracts text, analyzes it against rubrics, and generates feedback.

### Architecture

```
React Frontend    Express API    Python ML Service    PostgreSQL
     ↓                ↓                  ↓                ↓
  :5173            :3000             :8000          (Supabase)
```

- **Frontend**: React + TypeScript with Vite
- **Backend**: Express API with Supabase auth
- **ML**: Python pipeline for analysis (Vision models, LLMs, RAG)
- **Database**: PostgreSQL via Supabase

## Tech

- Frontend: React, TypeScript, Vite, React Router
- Backend: Express, TypeScript, Zod for validation
- Database: Supabase (PostgreSQL + Auth)
- ML: Python, Qwen2-VL, Phi-3, LangGraph, ChromaDB
- DevOps: Docker, Docker Compose, GitHub Actions

## Project Structure

```
mafa/
├── frontend/       React UI
├── backend/        Express API
├── ml-service/     Python ML pipeline
├── packages/       Shared types
├── docker-compose.yml
└── .env.example
```

## Quick Start

```bash
npm install
npm run dev          # Start frontend + backend
npm run dev:frontend # Frontend only
npm run dev:backend  # Backend only
```

### Environment

Copy `.env.example` to `.env` and add Supabase credentials.

## Docker

```bash
docker-compose up --build
```

## Development

- M1 ✅ Project setup
- M2 ⏳ Frontend UI (auth, dashboard, components)
- M3 📅 Backend API + database
- M4 📅 Tests and CI
- M5 📅 Deploy
- M6 📅 Polish

## Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
- [frontend/README.md](./frontend/README.md) - Frontend setup
- [backend/README.md](./backend/README.md) - API docs
