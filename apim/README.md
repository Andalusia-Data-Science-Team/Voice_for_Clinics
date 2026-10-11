# MedVoice APIM Configuration

This directory contains everything needed to deploy and configure Azure API Management (APIM) for the MedVoice application.

## Directory Structure

```
apim/
└── policies/
    ├── global.xml          ← Apply to "All APIs" → overrides all defaults
    ├── audio-analyze.xml   ← Apply to POST /audio/analyze
    ├── ws-transcribe.xml   ← Apply to GET /ws/transcribe (WebSocket)
    ├── patient-history.xml ← Apply to GET /patient/{code}/history
    └── admin.xml           ← Apply to GET /dashboard, /admin/usage, /admin/feedback
```

## Prerequisites

| Resource | Value |
|----------|-------|
| APIM tier | Standard or Premium (Premium required for WebSocket) |
| Entra App Registration | Client ID + Tenant ID |
| FastAPI backend | VNet-internal URL (e.g. `http://10.0.1.10:8586`) |

## Step-by-Step Deployment

### 1. Create Named Values

In Azure Portal → API Management → Named values, create:

| Name | Value | Secret? |
|------|-------|---------|
| `TENANT_ID` | Your Entra tenant GUID | No |
| `CLIENT_ID` | App Registration client_id | No |

### 2. Create the API

Portal → API Management → APIs → **+ Add API** → **HTTP**

| Field | Value |
|-------|-------|
| Display name | `MedVoice API` |
| Web service URL | `http://<fastapi-internal-ip>:8586` |
| API URL suffix | *(leave empty)* |
| Subscription required | **No** (JWT is the auth mechanism) |

### 3. Apply the Global Policy

Portal → API Management → APIs → **All APIs** → **Inbound processing** → Edit (pencil icon)

Paste the full contents of `policies/global.xml` → Save.

### 4. Apply Per-Operation Policies

For each operation below, navigate to it → Inbound processing → Edit → paste the policy:

| Operation | Policy file |
|-----------|-------------|
| `POST /audio/analyze` | `audio-analyze.xml` |
| `GET /ws/transcribe` | `ws-transcribe.xml` |
| `GET /patient/{code}/history` | `patient-history.xml` |
| `GET /dashboard` | `admin.xml` |
| `GET /admin/usage` | `admin.xml` |
| `GET /admin/feedback` | `admin.xml` |

### 5. Configure FastAPI

In `src/.env` (or Key Vault for production):

```env
APIM_ENABLED=true          # tells FastAPI to read X-User-* headers
DEV_CORS=false             # CORS is now handled by APIM
AZURE_KEY_VAULT_URL=...    # already set in Phase 0
```

### 6. Lock Down FastAPI

FastAPI must only accept connections from APIM, not the public internet:

- Deploy FastAPI inside the VNet (App Service VNet Integration, or AKS private cluster)
- APIM uses a private backend URL (e.g. `http://10.0.1.10:8586`)
- Add a Network Security Group rule: allow inbound 8586 only from APIM subnet

## Entra App Registration Setup

In Azure Portal → Microsoft Entra ID → App registrations → your app:

### Expose an API
- Application ID URI: `api://<CLIENT_ID>`
- Add scope: `access_as_user` (enabled for: Admins and users)

### App Roles (Manifest)
Add these to the `appRoles` array in the manifest:

```json
[
  {
    "allowedMemberTypes": ["User"],
    "displayName": "Doctor",
    "id": "<new-guid-1>",
    "isEnabled": true,
    "description": "Clinical doctor — can use all transcription and analysis features",
    "value": "Doctor"
  },
  {
    "allowedMemberTypes": ["User"],
    "displayName": "ClinicalAdmin",
    "id": "<new-guid-2>",
    "isEnabled": true,
    "description": "Clinical administrator — Doctor access + dashboard and audit log access",
    "value": "ClinicalAdmin"
  }
]
```

### Assign Roles to Users
Portal → Enterprise Applications → your app → Users and groups → + Add user/group

## Rate Limit Summary

| Operation | Limit | Policy file |
|-----------|-------|-------------|
| POST /audio/analyze | 10/min per user | audio-analyze.xml |
| WS /ws/transcribe | 20/min per user | ws-transcribe.xml |
| GET /patient/*/history | 30/min per user | patient-history.xml |
| GET /dashboard, /admin/* | ClinicalAdmin + 60/min | admin.xml |
| All others | 60/min per user | global.xml |

## Header Contract (APIM → FastAPI)

| Header | Source | Example |
|--------|--------|---------|
| `X-User-ID` | JWT `oid` claim | `a1b2c3d4-...` |
| `X-User-Email` | JWT `preferred_username` | `dr.ahmed@hospital.com` |
| `X-User-Name` | JWT `name` claim | `Dr. Ahmed Hassan` |
| `X-User-Roles` | JWT `roles` claim | `Doctor,ClinicalAdmin` |
| `X-Correlation-ID` | APIM-generated or passed through | `a1b2c3d4e5f6` |

> **Security note:** All `X-User-*` headers use `exists-action="override"` in the APIM policy, meaning a caller cannot spoof them — APIM always overwrites with values from the validated JWT.

## Verification

After deployment, test with curl from a machine that has a valid Entra token:

```bash
# Get token (Azure CLI)
TOKEN=$(az account get-access-token --resource api://<CLIENT_ID> --query accessToken -o tsv)

# Test authenticated route
curl -H "Authorization: Bearer $TOKEN" https://<apim-name>.azure-api.net/models

# Should return 403 for Doctor trying to access dashboard
curl -H "Authorization: Bearer $DOCTOR_TOKEN" https://<apim-name>.azure-api.net/dashboard

# Should return 200 for ClinicalAdmin
curl -H "Authorization: Bearer $ADMIN_TOKEN" https://<apim-name>.azure-api.net/dashboard
```
