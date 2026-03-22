# Running AuraAuth Locally

This project was built on Replit, which provides a PostgreSQL database,
environment variables, and an internal reverse proxy automatically.
When running locally you need to set those up yourself — this guide walks
you through every step.

---

## Prerequisites

Make sure the following are installed on your machine:

| Tool | Version | Install |
|------|---------|---------|
| Node.js | 20 or 22 | https://nodejs.org |
| pnpm | latest | `npm install -g pnpm` |
| PostgreSQL | 14+ | https://www.postgresql.org/download |
| Python | 3.11+ | https://www.python.org (only for the FastAPI backend) |
| Git | any | https://git-scm.com |

---

## 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>
pnpm install
```

---

## 2 — Create a local PostgreSQL database

```bash
# Log in to PostgreSQL
psql -U postgres

# Inside the psql prompt:
CREATE DATABASE auraauth;
\q
```

---

## 3 — Set up environment variables

### Express API server

```bash
cp artifacts/api-server/.env.example artifacts/api-server/.env
```

Open `artifacts/api-server/.env` and fill in:

```env
PORT=8080
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/auraauth
SESSION_SECRET=generate_a_random_string_here
```

Generate a session secret:
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### React frontend

```bash
cp artifacts/face-auth/.env.example artifacts/face-auth/.env
```

Open `artifacts/face-auth/.env` and fill in:

```env
PORT=5173
BASE_PATH=/
API_PORT=8080

# Optional — if not set the OTP will be shown on screen (dev mode)
VITE_EMAILJS_SERVICE_ID=your_service_id
VITE_EMAILJS_TEMPLATE_ID=your_template_id
VITE_EMAILJS_PUBLIC_KEY=your_public_key
```

> **EmailJS setup** (for real OTP emails):
> 1. Create a free account at https://www.emailjs.com
> 2. Add an email service (Gmail, Outlook, etc.)
> 3. Create a template with variables: `{{to_email}}`, `{{otp}}`, `{{app_name}}`
> 4. Copy your Service ID, Template ID, and Public Key into the `.env` file
>
> Without EmailJS the app still works — it shows the OTP code on the login screen.

---

## 4 — Push the database schema

```bash
pnpm --filter @workspace/db run push
```

This creates the `users` and `otp_verifications` tables in your local database.

---

## 5 — Run the project

You need **two terminals** running simultaneously.

### Terminal 1 — Express API server (port 8080)

```bash
# From the repo root
PORT=8080 pnpm --filter @workspace/api-server run dev
```

Or, if you have the `.env` file set up:
```bash
cd artifacts/api-server
# Using dotenv-cli (optional): npx dotenv -e .env -- pnpm run dev
PORT=8080 DATABASE_URL=... SESSION_SECRET=... pnpm run dev
```

You should see:
```
Server listening on port 8080
```

### Terminal 2 — React frontend (port 5173)

```bash
# From the repo root
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/face-auth run dev
```

You should see:
```
VITE v7.x.x  ready in xxx ms
➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

---

## 6 — How the API proxy works

In Replit, the platform automatically routes `/api/*` requests from the
frontend to the Express backend. Locally, Vite's built-in proxy does the
same job — it forwards every `/api/*` request from port `5173` to
`http://localhost:8080`. This is already configured in `vite.config.ts`;
you don't need to change anything.

---

## 7 — Optional: Run the FastAPI backend (Python)

The `backend/` folder contains a standalone FastAPI server. It is not
connected to the frontend yet but can be run independently.

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive API explorer.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `PORT environment variable is required` | Make sure you pass `PORT=...` before the dev command |
| `DATABASE_URL must be set` | Set `DATABASE_URL` in `artifacts/api-server/.env` or inline |
| API calls return 404 or network error | Make sure the API server is running on port `8080` before starting Vite |
| `Cannot find module '@workspace/...'` | Run `pnpm install` from the repo root |
| Database table does not exist | Run `pnpm --filter @workspace/db run push` |
| OTP not arriving | EmailJS not configured — check the dev mode OTP shown on screen instead |
| Face detection not working | Use HTTPS or `localhost` — browsers block camera on plain HTTP |
