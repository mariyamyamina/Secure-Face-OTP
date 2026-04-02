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

## Application: AuraAuth — Intelligent Anti-Spoof Face Authentication System

### Services
| Service | Port | Stack |
|---------|------|-------|
| Face Auth (frontend) | 19434 | React + Vite |
| API Server (backend) | 8080 | Express 5 + TypeScript |
| Anti-Spoof Service | 8000 | Python FastAPI + OpenCV |

### Authentication Pipeline
```
Camera → [Client-Side] → Anti-Spoof (JS) + Liveness Checks
       → [Server-Side] → POST /api/login-face
                       → Anti-Spoof Service (Python/OpenCV)  ← NEW
                       → Face Recognition (Euclidean distance)
                       → OTP Email Verification
```

### Anti-Spoofing (Server-Side Python — `artifacts/anti-spoof/main.py`)
Five OpenCV-based signals analysed per login attempt:
1. **FFT Periodic Pattern** — detects screen pixel-grid frequency artifacts
2. **LBP Texture Entropy** — real skin has richer micro-texture than screen renders
3. **Gradient Block Uniformity** — screen rendering produces suspiciously uniform local gradients
4. **Specular Highlight Detection** — screen glass produces concentrated glare hotspots
5. **YCbCr Skin Colour Ratio** — validates natural skin colour distribution

Spoof threshold: combined score ≥ 52 → FAKE → 403 rejected.

### Anti-Spoofing (Client-Side JS — `artifacts/face-auth/src/lib/antiSpoofing.ts`)
Five signals run every other detection tick (200 ms intervals):
- Glare / LBP texture / Colour naturalness / Temporal MAD / Motion CoV

### Liveness Detection (`artifacts/face-auth/src/lib/livenessDetector.ts`)
Four behavioural proofs required (hardened thresholds):
- Blink: full close+reopen cycle (single EAR drop not sufficient)
- Lip: 14 px open threshold (prevents detection-noise trigger)
- Head: 18 px nose-tip displacement (prevents phone-tilt trigger)
- Texture: MAD ≥ 2.0 in 5/8 samples

### API Endpoints
- `POST /api/register-face` - Register user with email, hashed password, and face descriptor
- `POST /api/login-face` - Pipeline: liveness check → server-side anti-spoof → face recognition
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
