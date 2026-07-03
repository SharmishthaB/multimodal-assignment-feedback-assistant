# Backend

Express API for MAFA.

## Setup

```bash
npm install
npm run dev      # Start dev server (port 3000)
npm run build    # Compile TypeScript
npm run start    # Start from compiled build
npm run lint     # ESLint
```

## Structure

- `src/index.ts` - Server entry point
- `src/routes/` - API route handlers
- `src/controllers/` - Business logic
- `src/middleware/` - Auth, validation
- `src/types/` - TypeScript interfaces

## API Endpoints (M3)

- `GET /api/health` - Server status
- `POST /api/auth/login` - Login
- `POST /api/auth/signup` - Signup
- `GET /api/assignments` - List assignments
- `POST /api/assignments` - Upload assignment
- `GET /api/assignments/:id` - Get assignment
- `GET /api/feedback/:assignmentId` - Get feedback

## Development

- Uses Express for HTTP server
- Zod for request validation
- Supabase for database and auth
- Axios for calling ML service
