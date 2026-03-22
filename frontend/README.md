# Frontend — AuraAuth Face Authentication System

The React frontend lives at `artifacts/face-auth/` inside the monorepo.
This folder exists for project orientation — the actual source code is not duplicated here.

## Why it's structured this way

This project uses a **pnpm monorepo** (pnpm workspaces). Moving the React
source into this folder would break cross-package imports like
`@workspace/api-client-react`, `@workspace/db`, and others that the frontend
depends on. The monorepo keeps all packages resolved correctly.

## Tech stack

- React 19
- Vite 7
- TypeScript
- Tailwind CSS 4
- Framer Motion
- wouter (routing)
- TanStack Query
- @vladmandic/face-api (face recognition)

## Source location

```
artifacts/face-auth/
├── src/
│   ├── pages/          # Home, Login, Register, Dashboard, AdminDashboard
│   ├── components/     # Shared UI components (Navbar, etc.)
│   ├── context/        # UserContext (auth state)
│   ├── lib/            # emailService, utils
│   └── App.tsx         # Router + providers
├── public/
├── index.html
├── vite.config.ts
└── package.json
```

## How to run the frontend

From the **workspace root**:

```bash
# Start the frontend dev server (runs on port 19434)
pnpm --filter @workspace/face-auth run dev

# Or start everything at once with the Replit workflows (recommended)
```

The dev server will be available at `http://localhost:19434`.

## Environment variables required

| Variable | Description |
|---|---|
| `PORT` | Dev server port (set to `19434`) |
| `BASE_PATH` | URL base path (set to `/`) |
| `VITE_EMAILJS_SERVICE_ID` | EmailJS service ID for OTP emails |
| `VITE_EMAILJS_TEMPLATE_ID` | EmailJS template ID |
| `VITE_EMAILJS_PUBLIC_KEY` | EmailJS public key |

## Build for production

```bash
pnpm --filter @workspace/face-auth run build
# Output: artifacts/face-auth/dist/public/
```
