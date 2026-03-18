# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/              # Deployable applications
│   ├── api-server/         # Express API server
│   └── face-auth/          # React + Vite frontend (Face Auth System)
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## Application: Intelligent Anti-Spoof Face Authentication System

### Features
- **Landing Page**: Hero section, feature cards, how-it-works steps
- **Registration Page**: Webcam capture, face-api.js face detection & 128D descriptor extraction
- **Login Page**: Placeholder page (OTP-based login coming soon)

### API Endpoints
- `POST /api/register-face` - Register user with email, hashed password, and face descriptor
- `POST /api/login-face` - Match face descriptor for authentication (euclidean distance threshold 0.6)
- `GET /api/healthz` - Health check

### Database Schema
- `users` table: id, email (unique), password (bcrypt-hashed), face_descriptor (JSON text), created_at

### Face Detection
- Library: `@vladmandic/face-api` (modern fork of face-api.js)
- Models loaded from CDN: `https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/`
- Backend forced to CPU for broader compatibility
- Models: TinyFaceDetector, FaceLandmark68Net, FaceRecognitionNet

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references.

- **Always typecheck from the root** — run `pnpm run typecheck`
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Packages

### `artifacts/face-auth` (`@workspace/face-auth`)

React + Vite frontend for the face auth system. Pages: Home, Register, Login.

Dependencies: `@vladmandic/face-api`, `react-webcam`, `framer-motion`, `axios` (via generated client)

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server with routes: `health.ts`, `auth.ts`.

Key deps: `bcrypt` (password hashing), `@workspace/db`, `@workspace/api-zod`

### `lib/db` (`@workspace/db`)

Drizzle ORM with PostgreSQL. Schema: `users` table.

- `pnpm --filter @workspace/db run push` — push schema to database

### `lib/api-spec` (`@workspace/api-spec`)

OpenAPI 3.1 spec (`openapi.yaml`) and Orval codegen config.

- `pnpm --filter @workspace/api-spec run codegen` — regenerate client + Zod schemas
