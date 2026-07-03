# Frontend

React + TypeScript web interface for MAFA.

## Setup

```bash
npm install
npm run dev      # Start Vite dev server (port 5173)
npm run build    # Production build
npm run lint     # ESLint
```

## Structure

- `src/pages/` - Route components (Dashboard, Assignment, etc)
- `src/components/` - Reusable React components
- `src/hooks/` - Custom hooks
- `src/types/` - TypeScript types
- `src/utils/` - Helper functions

## Development

- Uses Vite for fast dev builds
- React Router for navigation
- Axios for API calls
- Supabase client for auth

## Features (M2)

- Login/signup with Supabase Auth
- Dashboard with assignment list
- Assignment upload
- Results display
- Responsive design
