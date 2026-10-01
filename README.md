<h1 align="center">VoyageMesh<br />Multi-Agent Travel Planner</h1>

<p align="center">
  <strong>My personal travel-planning workspace, powered by coordinated AI agents.</strong>
</p>

<p align="center">
  <a href="#technology-stack"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python 3.12" /></a>
  <a href="#technology-stack"><img src="https://img.shields.io/badge/FastAPI-009688?style=flat&amp;logo=fastapi&amp;logoColor=white" alt="FastAPI" /></a>
  <a href="#technology-stack"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&amp;logo=langgraph&amp;logoColor=white" alt="LangGraph" /></a>
  <a href="#how-agents-are-discovered"><img src="https://img.shields.io/badge/A2A--style-HTTP-6B5B95?style=flat" alt="A2A-style HTTP" /></a>
  <a href="#technology-stack"><img src="https://img.shields.io/badge/MCP-Tools-555555?style=flat" alt="MCP tools" /></a>
  <a href="#configuration-and-optional-live-services"><img src="https://img.shields.io/badge/Groq-LLM-F55036?style=flat" alt="Groq LLM" /></a>
</p>

<p align="center">
  <a href="#technology-stack"><img src="https://img.shields.io/badge/React-18-149ECA?style=flat&amp;logo=react&amp;logoColor=white" alt="React 18" /></a>
  <a href="#technology-stack"><img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat&amp;logo=typescript&amp;logoColor=white" alt="TypeScript" /></a>
  <a href="#data-sources-and-rag"><img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat&amp;logo=postgresql&amp;logoColor=white" alt="PostgreSQL with pgvector" /></a>
  <a href="#technology-stack"><img src="https://img.shields.io/badge/Redis-7-DC382D?style=flat&amp;logo=redis&amp;logoColor=white" alt="Redis 7" /></a>
  <a href="#guardrails-and-security"><img src="https://img.shields.io/badge/Keycloak-4D9CBF?style=flat&amp;logo=keycloak&amp;logoColor=white" alt="Keycloak" /></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&amp;logo=docker&amp;logoColor=white" alt="Docker Compose" /></a>
</p>

<p align="center">
  I bring transport, accommodation and daily activities into one budget-checked plan,<br />
  with clear trade-offs, visible data sources and a workflow I can inspect.
</p>

---

## 📑 Table of Contents

- [Why I built it](#why-i-built-it)
- [How I use it](#how-i-use-it)
- [App preview](#app-preview)
- [Architecture](#architecture)
- [Planning workflow](#planning-workflow)
- [Request sequence](#request-sequence)
- [How agents are discovered](#how-agents-are-discovered)
- [Technology stack](#technology-stack)
- [Guardrails and security](#guardrails-and-security)
- [Data sources and RAG](#data-sources-and-rag)
- [Run locally](#run-locally)
  - [Prerequisites](#prerequisites)
  - [Prepare local demo settings](#1-prepare-local-demo-settings)
  - [Configure authentication and destination data](#2-add-the-local-authentication-and-destination-configuration)
  - [Sign in and plan a trip](#3-sign-in-and-plan-a-trip)
  - [Inspect or stop the stack](#4-inspect-or-stop-the-stack)
- [Call the API](#call-the-api)
- [Configuration and optional live services](#configuration-and-optional-live-services)
  - [Start live mode without API keys](#start-live-mode-without-api-keys)
  - [Add keys for live data](#add-keys-for-live-data)
- [Development, tests and evaluation](#development-tests-and-evaluation)
  - [Python and frontend development](#python-and-frontend-development)
  - [Tests without external services](#run-tests-without-external-services)
  - [Infrastructure integration tests](#infrastructure-integration-tests)
  - [Continuous integration](#continuous-integration)
- [Observability](#observability)
- [Repository layout](#repository-layout)
- [Validation and current limitations](#validation-and-current-limitations)
- [Explore the implementation](#explore-the-implementation)
- [License](#license)

---

## Why I built it

When I plan a trip, I want to answer a few practical questions together: **How do I get there? Where should I stay? What can I do each day? Does the whole trip fit my budget?** Comparing transport, accommodation and activities separately makes it easy to lose track of the total cost or put together an unrealistic schedule.

I use VoyageMesh to bring those decisions into one workflow. I enter my route, dates, budget and preferences, then review a structured plan with ranked travel options, accommodation, a daily itinerary and an itemised budget. I can compare the trade-offs and see which information is simulated, estimated, cached or live.

The project also gives me a practical way to explore how a multi-agent system is engineered: independent services, explicit contracts, controlled tool access, authentication, failure handling, tracing and repeatable evaluation. The planning workflow makes the decisions in code; the language model explains the results.

> **Current scope:** a local planning app with an offline demo mode and optional live searches through Duffel, LiteAPI, Geoapify and Open-Meteo. Paid-source adapters need production credentials; their HTTP contracts are tested offline, but real-account verification is still pending. Rail/bus integration needs Omio partner documentation. VoyageMesh does not book tickets, reserve rooms or process payments.

## How I use it

For a short trip such as **Nuremberg → Prague**, I provide:

| Input | Example |
| --- | --- |
| Dates | A departure date and a return date three nights later |
| Travellers | 1 |
| Maximum budget | EUR 350 for the whole trip |
| Accommodation | Hostel or budget hotel preference |
| Transport | Any supported mode, or a specific preference |
| Interests | History, architecture and local food |
| Constraints | At most 8 hours of transport and 2 transfers |
| Ranking | Balanced, cheapest, fastest or fewest transfers |
| Accessibility | Optional accessibility preferences |

I review the recommended transport and stay, compare up to five alternatives for each, and check the itinerary and budget before making my own travel decisions. The budget includes transport, accommodation, priced activities and explicit estimates for food and local transport.

The result also includes weather considerations, trade-off explanations, source labels, available retrieval timestamps, confidence, completeness and any unavailable sections. The UI presents these through **Overview, Transport, Accommodation, Itinerary, Budget, Sources, Agent activity and Observability** tabs.

Changing my budget or preferences produces another plan. Mock mode can reuse Redis's cached result; live mode fetches fresh provider data for each new search. If a dependency fails, the response identifies the gap and returns the usable sections where possible.

## App preview

The trip form below shows the current interface. The expandable result shows an earlier **mock-mode demo**, with simulated prices; live quotes require the provider keys described in [live data setup](#add-keys-for-live-data). See [Observability](#observability) for a screenshot of the request diagnostics.

<p align="center">
  <img src="images/three.png" alt="VoyageMesh trip form with route, dates, adult travellers, guest nationality, budget, ranking and accessibility preferences" width="960" />
  <br />
  <em>Set my route, dates, budget and preferences in one place, including guest nationality for live hotel searches.</em>
</p>

<details>
<summary><strong>View the completed demo plan and agent execution summary</strong></summary>

<p align="center">
  <img src="images/two.png" alt="Demo result showing completed specialist agents, plan status and a simulated budget summary" width="960" />
  <br />
  <em>Review the completed demo plan, specialist-agent status and budget trade-offs.</em>
</p>

</details>

## Architecture

The **LangGraph orchestrator runs inside the FastAPI gateway process**. Each specialist agent and each MCP server runs as a separate service. The browser talks to the gateway; the orchestrator delegates over HTTP; specialists obtain travel data through MCP tools.

```mermaid
flowchart TB
    UI["React frontend"] -->|"Trip request + access token"| GW
    UI <-->|"OIDC login with PKCE"| KC["Keycloak"]

    subgraph Gateway["API gateway process"]
        GW["FastAPI: authentication, rate limits, idempotency"]
        ORCH["LangGraph orchestrator"]
        GW --> ORCH
    end

    GW -.->|"Validate JWT using public keys"| KC
    ORCH -->|"A2A-style HTTP"| TA["Transport Agent"]
    ORCH -->|"A2A-style HTTP"| SA["Stay Agent"]
    ORCH -->|"A2A-style HTTP"| IA["Itinerary Agent"]
    TA -->|MCP| TM["Transport MCP"]
    SA -->|MCP| LM["Lodging MCP"]
    IA -->|MCP| DM["Destination MCP"]
    TM --> TP["Mock transport or Duffel flights"]
    LM --> LP["Mock accommodation or LiteAPI"]
    DM --> DEST["Fixtures or Geoapify + Open-Meteo"]
    DM -.->|"Available knowledge-retrieval tool"| PG[("PostgreSQL + pgvector")]
    ORCH <-->|"Mock mode only"| RD[("Redis plan cache")]
    GW -->|"Persist request and plan"| PG
    TA & SA & IA --> LLM["Mock LLM or Groq narration"]
```

The gateway also uses Redis for rate limiting and idempotency. OpenTelemetry, Jaeger, Prometheus and Grafana provide traces, metrics and dashboards alongside the application.

| Component | Responsibility |
| --- | --- |
| Orchestrator | Validate and normalise the request, check the cache, delegate tasks, combine results, calculate the budget and control replanning |
| Transport Agent | Search ground/air options through MCP, consume deterministic rankings and explain the recommendation |
| Stay Agent | Search accommodation through MCP, compare total stay costs and explain the recommendation |
| Itinerary Agent | Gather attractions and weather, build a daily schedule in code and validate it through MCP |
| MCP servers | Expose named, schema-validated tools and call the relevant provider or domain functions |
| Shared agent harness | Bound execution, validate structured results, record actions, handle retries and generate explanatory text |

**A2A and MCP serve different purposes:** A2A-style communication delegates a task to another agent; MCP invokes a tool. Here, A2A is a project-specific Agent Card/task/artifact HTTP contract, isolated behind an `AgentClient` interface. It is not a claim of conformance with every version of an external A2A standard.

## Planning workflow

Transport and accommodation searches run **in parallel**. The itinerary stage follows their completion. Routing and replanning are deterministic LangGraph decisions; an LLM does not select the next graph node.

```mermaid
flowchart TD
    INPUT(["Trip form or API request"]) --> API["Authenticate, rate-limit and check idempotency"]
    API --> VALIDATE{"Input valid?"}
    VALIDATE -->|No| REJECT["Return validation failure"]
    VALIDATE -->|Yes| NORMALIZE["Normalise trip preferences"]
    NORMALIZE --> CACHE{"Mock mode with a fresh cached plan?"}
    CACHE -->|Yes| CACHED["Reuse cached demo plan"]
    CACHE -->|"Miss or live mode"| DISCOVER["Configured agent URLs + Agent Card skill checks"]

    subgraph SEARCH["Parallel searches through specialist agents and MCP"]
        T["Transport Agent"] --> TM["Transport MCP: search and rank"]
        S["Stay Agent"] --> SM["Lodging MCP: search and rank"]
    end

    DISCOVER --> T
    DISCOVER --> S
    TM --> JOIN["Collect transport and stay results or warnings"]
    SM --> JOIN
    JOIN --> I["Itinerary Agent + Destination MCP: places and weather"]
    I --> COST["Calculate budget in Python"]
    COST --> CHECK{"Complete budget exceeds limit and replans remain?"}
    CHECK -->|Yes| REPLAN["Tighten accommodation allowance"]
    REPLAN --> DISCOVER
    CHECK -->|"No, or costs are incomplete"| FINAL["Assemble plan and run output guardrails"]
    FINAL --> CACHEWRITE["Cache usable results in mock mode only"]
    CACHED --> SAVE["Gateway attempts to persist the plan"]
    CACHEWRITE --> SAVE
    SAVE --> OUT(["UI: recommendations, itinerary, budget, sources and warnings"])

    classDef entry fill:#0f172a,color:#ffffff,stroke:#64748b
    classDef agent fill:#dbeafe,color:#172554,stroke:#2563eb
    classDef tool fill:#ccfbf1,color:#134e4a,stroke:#0d9488
    classDef decision fill:#fef3c7,color:#78350f,stroke:#d97706
    class INPUT,OUT entry
    class T,S,I agent
    class TM,SM tool
    class VALIDATE,CACHE,CHECK decision
```

Demo defaults allow at most **3 replans** and a **45-second planning deadline**. The live Compose override uses **0 replans** and a **90-second deadline** to avoid repeated metered searches. Incomplete budgets finish with warnings instead of repeating unavailable searches. The user's budget is not increased to manufacture a successful result. Missing services or an exhausted replan budget can produce a partial plan; missing viable transport can produce `no_viable_plan`. Blocking output checks produce `failed`.

## Request sequence

This sequence shows a new live search with authentication enabled. Each delegation fetches and validates the specialist's card before sending a task. The shared MCP participant represents three separate tool servers; only those servers call travel providers.

```mermaid
sequenceDiagram
    autonumber
    actor U as Traveller
    participant UI as React frontend
    participant K as Keycloak
    participant O as Gateway + LangGraph
    participant T as Transport Agent
    participant S as Stay Agent
    participant I as Itinerary Agent
    participant M as MCP servers
    participant P as Travel providers
    participant DB as Redis / PostgreSQL

    U->>UI: Sign in and enter trip preferences
    UI->>K: Authorization Code + PKCE
    K-->>UI: User access token
    UI->>O: POST /api/v1/trips with bearer token
    O->>O: Validate JWT, role and input
    O->>DB: Check request idempotency
    DB-->>O: New request
    Note over O,DB: Live mode bypasses the plan cache

    par Transport search
        O->>T: GET /.well-known/agent-card.json
        T-->>O: Card advertising plan_transport
        O->>O: Verify required skill
        O->>K: Obtain or reuse service token
        K-->>O: Token with agent:invoke role
        O->>T: POST /a2a/tasks with service token
        T->>M: Search, validate and rank transport
        M->>P: Duffel flight search using mounted key
        P-->>M: Offers or provider error
        M-->>T: Typed offers or unavailable result
        T-->>O: Transport artifact and explanation
    and Accommodation search
        O->>S: GET /.well-known/agent-card.json
        S-->>O: Card advertising plan_accommodation
        O->>O: Verify skill and obtain service token
        O->>S: POST /a2a/tasks with service token
        S->>M: Search, validate and rank stays
        M->>P: LiteAPI hotel rates using mounted key
        P-->>M: Rates or provider error
        M-->>S: Typed stays or unavailable result
        S-->>O: Accommodation artifact and explanation
    end

    O->>I: GET /.well-known/agent-card.json
    I-->>O: Card advertising plan_itinerary
    O->>O: Verify skill and obtain service token
    O->>I: POST /a2a/tasks with preferences and stay location
    I->>M: Get attractions and weather, then validate daily schedules
    M->>P: Geoapify places and keyless Open-Meteo weather
    P-->>M: Available destination data
    M-->>I: Destination data and validation results
    I-->>O: Itinerary artifact and explanation
    O->>O: Validate artifacts and calculate budget
    opt Complete budget is over limit and replanning is enabled
        O->>O: Repeat delegation with tighter stay allowance
    end
    O->>O: Assemble plan and check output
    O->>DB: Attempt to persist request and plan
    O-->>UI: TripPlan with status, sources and warnings
    UI-->>U: Show recommendations, itinerary and budget
```

LLM narration occurs inside the specialist agents after their structured results have been computed. The orchestrator does not call travel-provider APIs directly.

When a required provider key is missing, its MCP server returns an explicit unavailable result **before making the authenticated API request**. Available sections, such as keyless weather, can still be returned; missing quotes are never replaced with invented live prices.

## How agents are discovered

Discovery has two parts: **configuration supplies the address; the Agent Card describes the capabilities**.

| Agent | Configuration key | Docker network address | Required skill |
| --- | --- | --- | --- |
| Transport | `A2A_TRANSPORT_AGENT_URL` | `http://transport-agent:8010` | `plan_transport` |
| Stay | `A2A_STAY_AGENT_URL` | `http://stay-agent:8020` | `plan_accommodation` |
| Itinerary | `A2A_ITINERARY_AGENT_URL` | `http://itinerary-agent:8030` | `plan_itinerary` |

For each task, the client:

1. Fetches `GET <configured-address>/.well-known/agent-card.json`.
2. Parses the card into the shared `AgentCard` schema.
3. Checks that it advertises the required skill.
4. Submits an `AgentTask` to `POST <configured-address>/a2a/tasks` with service credentials and trace context.
5. Parses the returned artifact and checks its task and correlation IDs. The orchestrator then validates the specialist result against its known schema.

Cards contain the agent's identity, version, URL, skills, input/output schemas, capabilities, authentication scheme and required roles. The card endpoint is available without a bearer token inside the service network; task submission is authenticated when service auth is configured.

**Agent registration is explicit.** The workflow has three configured clients; it does not scan the network, use a central agent registry or ask an LLM to discover new services. Publishing another card alone does not add an agent to the workflow. Card schemas are descriptive; compatibility checks use shared models and the required skill name, rather than negotiating arbitrary schemas or protocol versions.

See [`submit_verified()` and `HttpAgentClient`](packages/vm_harness/a2a_client.py), [dependency wiring](apps/vm_api_gateway/services.py) and the [Agent Card contract](packages/vm_contracts/a2a.py).

## Technology stack

| Area | Technologies | Use in this project |
| --- | --- | --- |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic v2 | Async services, API routes and validated contracts |
| Workflow | LangGraph | Typed state, parallel execution, conditional edges and replanning |
| Agent communication | HTTPX, project-specific A2A contracts | Agent Cards, authenticated tasks and structured artifacts |
| Tool access | MCP Python SDK / FastMCP | Three tool servers using Streamable HTTP |
| Language model | Groq SDK, mock provider | Structured explanations with a deterministic test alternative |
| Frontend | React 18, TypeScript, Vite, nginx | Trip form, OIDC login, result tabs and API proxy |
| Persistence | PostgreSQL 16, SQLAlchemy, asyncpg, Alembic | Stored requests/plans, migrations and evaluation storage |
| Retrieval | pgvector, local 384-dimensional hashing embedder | Curated destination-knowledge ingestion and retrieval |
| Cache and coordination | Redis 7 | Plan cache, idempotency, locking and rate limiting |
| Identity | Keycloak, OIDC + PKCE, PyJWT | Browser login, JWT validation and service credentials |
| Observability | OpenTelemetry, Jaeger, Prometheus, Grafana | Distributed traces, metrics and eight dashboards |
| Packaging and runtime | uv, Docker, Docker Compose | Locked Python dependencies and local service profiles |
| Quality and CI | pytest, Ruff, mypy, Bandit, pip-audit, Gitleaks, Hadolint, Trivy, GitHub Actions | Tests, evaluation, static checks, builds and security scans |

Python dependencies are locked in [`uv.lock`](uv.lock); frontend dependencies are locked in [`package-lock.json`](apps/frontend/package-lock.json).

## Guardrails and security

Guardrails are implemented in code at several boundaries. They reduce specific risks; they are not a guarantee that every unsafe input or inaccurate narrative will be detected.

| Boundary | Implemented controls |
| --- | --- |
| Trip input | Typed fields; positive budget; supported currencies/enums; consistent dates; maximum trip length, traveller count and text sizes; restricted place-name syntax |
| Prompt injection | Pattern screening of places, notes and interests for instruction overrides, role reassignment, prompt probes and selected exfiltration patterns |
| Agent execution | Fixed tool-selection logic; bounded task/request execution; tool and LLM call budgets; retries and structured failure results |
| Tools and providers | Named tools with Pydantic arguments; read-only operations; provider hostname allowlists; rejection of literal IPs, URL credentials and redirects; timeouts, response-size caps and circuit breakers |
| Money and schedules | Deterministic scoring and `Decimal`-based money; currency and budget-sum validation; activity ordering, overlap and travel-gap checks |
| Final output | Pattern checks for booking claims and credential-shaped text in the combined narrative; warnings for selected source-label/timestamp and status inconsistencies |
| Identity and access | JWT signature, issuer, audience, expiry and required-claim checks; role checks; optional Keycloak service authentication requiring `agent:invoke` |
| API resilience | Request-size checks, Redis rate limiting, idempotency keys, trace IDs and sanitised error responses |
| Knowledge retrieval | Curated corpus, rejection of price-shaped text during ingestion, citation metadata and a warning that retrieved text is untrusted |
| Secrets | Runtime file loading, redaction helpers, ignored secret paths and read-only Docker secret mounts |

The model is used for narration. It is not given tools for arbitrary shell commands, SQL, filesystem access or user-selected outbound URLs. Authentication, arithmetic and workflow branching remain outside model control. Action records contain concise summaries rather than private chain-of-thought.

Default planning limits are **3 replans**, **8 logical tool calls per agent**, **6 LLM calls per agent**, **2 A2A retries**, **2 MCP retries**, a **25-second agent task deadline** and a **45-second orchestration deadline**. These are application defaults; the overall HTTP request also includes gateway work.

The base Compose setup allows local authentication bypass. The authenticated setup below explicitly enables user and A2A authentication. MCP services remain on the internal Docker network and do not have a separate authentication layer in the current wiring. The development realm and published infrastructure ports are intended for a trusted development machine.

Implementation: [input/output checks](packages/vm_guardrails/), [safe provider HTTP client](packages/vm_resilience/safe_http.py), [authentication](packages/vm_auth/) and [agent harness](packages/vm_harness/).

## Data sources and RAG

| Information | Current source | Meaning |
| --- | --- | --- |
| Flights | Duffel v2 in `live`/`auto`; mock provider in `mock` | Production quotes, priced for all adults and both journeys when a return is requested |
| Rail and bus | Mock provider in `mock`; unavailable in `live`/`auto` | Omio partner specification and credentials are needed before integration |
| Accommodation | LiteAPI v3 in `live`/`auto`; mock provider in `mock` | Production rates for one room, all adults and the full stay, including disclosed mandatory property fees |
| Attractions | Geoapify Places in `live`/`auto`; curated fixtures in `mock` | Live location listings; admission prices, opening hours and accessibility remain unverified |
| Weather | Open-Meteo in `live`/`auto`; synthetic conditions in `mock` | `LIVE` forecast where available; unavailable outside coverage or on errors |
| Geocoding | Open-Meteo in `live`/`auto`; curated city centres in `mock` | Provider-resolved coordinates and a warning identifying the resolved place |
| Budget and rankings | Python domain functions | Computed values; food/local-transport allowances are estimates |
| Destination knowledge | Curated corpus in PostgreSQL/pgvector | Citation-bearing reference text, not live prices or availability |

The RAG pipeline supports ingestion, chunking and vector retrieval through `retrieve_destination_knowledge`. Its embedder is a lightweight local hashing implementation, not a pretrained semantic embedding model. **The current Itinerary Agent does not invoke that knowledge tool during normal planning.** RAG is an implemented supporting capability, with automatic itinerary integration left as follow-up work.

Choosing Groq changes narration independently of travel data. `mock` is the only mode that supplies simulated offers. Both `live` and `auto` use external adapters and return unavailable data when a key is missing, access is refused or a query fails. Sandbox flight/hotel quotes are refused. No currency conversion is guessed. Unpriced admissions from live place listings leave the budget incomplete; food and local transport remain explicitly estimated.

See [live provider setup](#add-keys-for-live-data) for credential paths and activation. Live searches bypass the Redis plan cache so an old mock plan or expired quote cannot be reused as a fresh result. Saved trips remain historical snapshots.

## Run locally

### Prerequisites

- Git and a checkout of this repository.
- Docker Desktop with the Linux container engine running and Docker Compose v2 available.
- PowerShell for the commands below. Run them from the repository root.
- Internet access for initial image/dependency downloads. The demo below uses mock travel data and a mock LLM afterward.
- For development outside Docker: Python **3.12**, **uv**, and Node.js **22** with npm for frontend work.

### 1. Prepare local demo settings

This guide starts the full browser experience with Keycloak login, authenticated agent calls, Redis, PostgreSQL and observability. No Groq key is required. Local setup files go in the already ignored `.tmp/` directory.

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null

if (-not (Test-Path .tmp/groq-placeholder.key)) {
    Set-Content .tmp/groq-placeholder.key -Value "unused-in-mock-mode" -Encoding ascii
}
if (-not (Test-Path .tmp/orchestrator-client.key)) {
    Set-Content .tmp/orchestrator-client.key -Value "CHANGE_ME_ORCHESTRATOR_SECRET" -Encoding ascii
}

$env:ENVIRONMENT = "local"
$env:LLM_PROVIDER = "mock"
$env:PROVIDER_MODE = "mock"
$env:OTEL_ENABLED = "true"
$env:GROQ_API_KEY_FILE_HOST = "./.tmp/groq-placeholder.key"
```

Compose mounts a Groq secret file even in mock mode, so an existing placeholder file is necessary. The service-client value above matches the **development-only placeholder** in the imported realm. It is not a production credential. An existing customised realm must use its matching client secret instead.

### 2. Add the local authentication and destination configuration

The base Compose file does not forward every authentication/provider setting to every service. Create this override so browser-facing token issuers, container-internal Keycloak endpoints and destination database access are explicit:

```powershell
@'
x-local-auth: &local-auth
  AUTH_ISSUER: http://localhost:8080/realms/voyagemesh
  AUTH_JWKS_URL: http://keycloak:8080/realms/voyagemesh/protocol/openid-connect/certs
  AUTH_AUDIENCE: voyagemesh-api
  AUTH_SERVICE_AUTH: keycloak

secrets:
  orchestrator_client_secret:
    file: ./.tmp/orchestrator-client.key

services:
  keycloak:
    environment:
      KC_HOSTNAME: http://localhost:8080
  api-gateway:
    environment:
      <<: *local-auth
      AUTH_DEV_INSECURE_ALLOW_UNAUTHENTICATED: "false"
      AUTH_SERVICE_CLIENT_ID: voyagemesh-orchestrator
      AUTH_SERVICE_CLIENT_SECRET_FILE: /run/secrets/orchestrator_client_secret
      AUTH_TOKEN_URL: http://keycloak:8080/realms/voyagemesh/protocol/openid-connect/token
    secrets:
      - orchestrator_client_secret
  transport-agent:
    environment:
      <<: *local-auth
  stay-agent:
    environment:
      <<: *local-auth
  itinerary-agent:
    environment:
      <<: *local-auth
  destination-mcp:
    environment:
      DB_URL: postgresql+asyncpg://voyagemesh:voyagemesh@postgres:5432/voyagemesh
      PROVIDER_MODE: ${PROVIDER_MODE:-mock}
    depends_on:
      seed:
        condition: service_completed_successfully
'@ | Set-Content .tmp/compose.local.yaml -Encoding utf8

$composeFiles = @("-f", "docker-compose.yml", "-f", ".tmp/compose.local.yaml")
docker compose @composeFiles --profile full config --quiet
docker compose @composeFiles --profile full up -d --build
docker compose @composeFiles --profile full ps --all
```

Keep this PowerShell session open: later commands reuse `$composeFiles` and the environment settings. On a new session, restore them before running Compose commands. Migrations and corpus seeding are one-off services; an exit code of `0` is expected for those services. Wait until the application services are healthy and the Keycloak login page is available.

### 3. Sign in and plan a trip

Open **[http://localhost:3000](http://localhost:3000)** and choose **Sign in with Keycloak**.

| Local development account | Password | Application roles |
| --- | --- | --- |
| `traveller` | `traveller` | Traveller |
| `evaluator` | `evaluator` | Traveller, evaluator |
| `admin` | `admin` | Traveller, evaluator, admin |

Use Nuremberg → Prague with a hostel preference, one traveller, a EUR 350 budget and three nights as a starting scenario. Review the source labels alongside the recommendations.

| Service | Local address |
| --- | --- |
| Frontend | [localhost:3000](http://localhost:3000) |
| API / Swagger UI | [localhost:8000/docs](http://localhost:8000/docs) |
| Gateway health | [localhost:8000/health/ready](http://localhost:8000/health/ready) |
| Keycloak | [localhost:8080](http://localhost:8080) |
| Grafana | [localhost:3001](http://localhost:3001), local login `admin` / `admin` |
| Prometheus | [localhost:9090](http://localhost:9090) |
| Jaeger | [localhost:16686](http://localhost:16686) |
| PostgreSQL / Redis | `localhost:5433` / `localhost:6380` |

Agent and MCP ports are internal to the Docker network; the base Compose file does not publish them to the host.

### 4. Inspect or stop the stack

```powershell
docker compose @composeFiles --profile full logs --tail 100 api-gateway
docker compose @composeFiles --profile full logs --tail 100 transport-agent stay-agent itinerary-agent
docker compose @composeFiles --profile full down
```

Stopping this way retains named database, Grafana and Prometheus volumes. Redis has persistence disabled in this local setup, so its cached state is temporary.

Profiles are `core` (application and stores), `auth` (Keycloak and supporting stores/jobs), `observability` (monitoring and supporting stores/jobs), `full` (all services) and `test` (PostgreSQL/Redis). Combine profiles when needed; `auth` or `observability` alone does not start the frontend and agents. Starting `full` alone does not enforce authentication; the override above supplies that wiring.

## Call the API

Planning is synchronous: `POST /api/v1/trips` returns the final `TripPlan`. `GET /api/v1/trips/{trip_id}` reads a persisted result; it is not a background-job progress endpoint.

The following uses the realm's local development password grant for a CLI demonstration. Browser login uses Authorization Code + PKCE.

```powershell
$gateway = "http://localhost:8000"
$tokenResponse = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8080/realms/voyagemesh/protocol/openid-connect/token" `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{
        grant_type = "password"
        client_id = "voyagemesh-frontend"
        username = "traveller"
        password = "traveller"
    }

$headers = @{
    Authorization = "Bearer $($tokenResponse.access_token)"
    "Idempotency-Key" = [guid]::NewGuid().ToString()
}
$departure = (Get-Date).Date.AddDays(7)
$trip = @{
    origin = "Nuremberg"
    destination = "Prague"
    departure_date = $departure.ToString("yyyy-MM-dd")
    return_date = $departure.AddDays(3).ToString("yyyy-MM-dd")
    travellers = 1
    max_budget = @{ amount = "350.00"; currency = "EUR" }
    accommodation_preference = "hostel"
    transport_preference = "any"
    interests = @("history", "architecture", "local food")
    max_transport_duration_hours = 8
    max_transfers = 2
    ranking_strategy = "balanced"
}

$plan = Invoke-RestMethod -Method Post -Uri "$gateway/api/v1/trips" `
    -Headers $headers -ContentType "application/json" `
    -Body ($trip | ConvertTo-Json -Depth 8)

$plan | Select-Object trip_id, status, cache_status, within_budget, completeness_percent
$plan.budget | Select-Object total, remaining, overspend, status
$plan.data_sources | Format-Table component, source_name, origin
```

Use `$plan | ConvertTo-Json -Depth 30` to inspect the full response. Retrieve the saved plan with:

```powershell
Invoke-RestMethod -Uri "$gateway/api/v1/trips/$($plan.trip_id)" -Headers $headers
```

Repeating a request with the same `Idempotency-Key` replays its stored response. To demonstrate the **plan cache**, send the same trip with a **new** idempotency key and inspect `cache_status`.

| Plan status | Meaning |
| --- | --- |
| `complete` | Required sections produced without a degraded dependency |
| `partial` | Usable output with missing data, an unmet budget or another reported limitation |
| `no_viable_plan` | No usable transport recommendation could be produced |
| `failed` | Validation, a blocking guardrail or a planning failure prevented a usable plan |

HTTP success does not imply a complete trip. Inspect status, warnings, budget and completeness. API errors use a structured contract; examples include `401`, `403`, `409`, `422` and `429` for authentication, authorisation, an in-flight duplicate, invalid input and rate limiting.

## Configuration and optional live services

[`settings.py`](packages/vm_config/settings.py) defines defaults and validation. [`.env.example`](.env.example) lists application settings. A host `.env` file does not automatically inject every setting into Docker containers; Compose must explicitly forward it.

| Setting | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `mock` for deterministic explanations; `groq` for hosted narration |
| `GROQ_API_KEY_FILE_HOST` | Host file used as the Compose secret source |
| `GROQ_API_KEY_FILE` | Runtime key path; `/run/secrets/groq_api_key` inside agents |
| `GROQ_MODEL` | Model identifier read by each agent's LLM settings |
| `PROVIDER_MODE` | `mock` for fixtures; `auto`/`live` for external APIs, with unavailable results on failure |
| `PROVIDER_DUFFEL_API_KEY_FILE` | File containing a Duffel live access token |
| `PROVIDER_LITEAPI_API_KEY_FILE` | File containing a LiteAPI production key |
| `PROVIDER_GEOAPIFY_API_KEY_FILE` | File containing a Geoapify key |
| `A2A_*_AGENT_URL` / `MCP_*_URL` | Agent and tool-server addresses |
| `AUTH_ISSUER` / `AUTH_JWKS_URL` | Expected token issuer and reachable public-key endpoint |
| `AUTH_SERVICE_AUTH` | Keycloak service tokens or development shared-secret mode |
| `DB_URL` / `REDIS_URL` | Database and cache connections |
| `OTEL_ENABLED` | Enable trace instrumentation/export |
| `GATEWAY_HOST_PORT` | Override the gateway's published host port |

**Enable Groq:** create `apikeys/api.key` locally containing only your own key, then use the same Compose files:

```powershell
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY_FILE_HOST = "./apikeys/api.key"
docker compose @composeFiles --profile full up -d
```

The key is read at runtime and mounted read-only into agents. Do not put it in the README, frontend, Dockerfile or Git. The base Compose file uses the application's default Groq model; to choose another, add `GROQ_MODEL` to each agent's environment in the local override. Model access depends on your Groq account. Hosted calls require network access and are subject to provider limits.

### Start live mode without API keys

Complete the [local setup](#run-locally) first, including the local authentication override and mock-Groq placeholder. You can then start the live stack **before obtaining any travel-provider keys**:

```powershell
New-Item -ItemType Directory -Force apikeys/duffel, apikeys/liteapi, apikeys/geoapify | Out-Null
docker compose @composeFiles -f docker-compose.live.yml --profile full up -d --build
docker compose @composeFiles -f docker-compose.live.yml --profile full ps
```

Add `docker-compose.live.yml` **last**, after the local authentication override. It explicitly sets live mode even when the demo environment says `PROVIDER_MODE=mock`.

The UI at [localhost:3000](http://localhost:3000) and login still work without provider keys. A trip search reports unavailable flight, hotel and attraction data, with missing-key messages in Overview. Expect `no_viable_plan` when no transport can be priced. This is a working application with unavailable travel data, not a complete trip. Open-Meteo can still supply geocoding and weather without a key when its public API is reachable and the dates are within forecast coverage. No mock prices are substituted in live mode.

### Add keys for live data

Save each credential in its own local file, relative to the repository root:

| Provider | Exact file to create | Required credential | Enables |
| --- | --- | --- | --- |
| Duffel | `apikeys/duffel/api.key` | Live access token | Flight prices and outbound/return journeys |
| LiteAPI / Nuitee | `apikeys/liteapi/api.key` | Production API key | Hotel availability and rates |
| Geoapify | `apikeys/geoapify/api.key` | API key | Attraction and place discovery |

Each file must contain **only the key**, without quotes, `KEY=` prefixes or JSON. Use plain UTF-8 text without a byte-order mark, for example by creating the file in VS Code. These directories are excluded from Git and Docker build contexts. The live override mounts each directory read-only into only its corresponding MCP server.

Keys can be added independently. Once this live stack is running, adding or replacing an `api.key` file in an already mounted directory is picked up on the **next new trip search**; no code edit, image rebuild or container restart is required. If you change the host directory itself, rerun Compose to update the mount. A saved trip is a historical result: submit a new request to fetch new prices.

In the UI, select `flight` (or `any`), choose future dates and supply the guest's actual nationality as a two-letter code for hotel rates. Duffel and LiteAPI sandbox credentials are refused. Groq is optional and affects narration only. Rail/bus search remains unavailable in live mode until the Omio partner API is integrated.

If you launch with `.tmp/app-launch.env`, keep that argument when recreating the stack:

```powershell
docker compose --env-file .tmp/app-launch.env -f docker-compose.yml -f .tmp/compose.local.yaml -f docker-compose.live.yml --profile full up -d --build
```

For direct Python runs, set `PROVIDER_MODE=live` and point `PROVIDER_DUFFEL_API_KEY_FILE`, `PROVIDER_LITEAPI_API_KEY_FILE` and `PROVIDER_GEOAPIFY_API_KEY_FILE` at your local key files. Use the longer request limits in [`docker-compose.live.yml`](docker-compose.live.yml) as a reference. In Docker, change the host directories through `DUFFEL_SECRETS_DIR`, `LITEAPI_SECRETS_DIR` and `GEOAPIFY_SECRETS_DIR`, then recreate the affected MCP containers to apply the mount changes.

For live searches, select **flight** or **any**, use future dates and enter the hotel guest nationality as an uppercase two-letter country code, such as `DE` or `IN`. This version treats travellers as adults and searches one room for the whole group. Mixed nationalities, children and multiple-room occupancy are not supported. If a key is rejected, check that it belongs to the correct provider and production environment; missing keys and upstream failures appear in the plan warnings.

**Port conflict:** set `$env:GATEWAY_HOST_PORT = "8001"` before `up`, then use `http://localhost:8001` for direct API calls. The frontend's internal proxy remains pointed at container port `8000`.

**Login troubleshooting:** use `localhost` consistently. The expected issuer is the browser-facing `http://localhost:8080/realms/voyagemesh`; JWKS and service-token requests use the Docker hostname `keycloak`. Check Keycloak logs and the imported realm if login fails. Existing realms are not overwritten by a fresh import file.

## Development, tests and evaluation

### Python and frontend development

```powershell
uv python install 3.12
uv sync --locked --extra observability
uv run --extra observability ruff check .
uv run --extra observability ruff format --check .
uv run --extra observability mypy packages apps agents mcp_servers evaluations
```

For frontend changes, run `npm ci` and `npm run build` from `apps/frontend`. Vite also provides `npm run dev`; its default port is `3000`, so stop the Docker frontend or choose another development port first. Authentication settings are compiled into the Vite bundle, rather than read from container environment variables at runtime.

### Run tests without external services

```powershell
$env:OTEL_ENABLED = "false"
uv run pytest -q -m "not requires_docker and not requires_groq and not requires_network"
uv run python -m evaluations.run --suite regression
```

This includes unit, contract, security, evaluation and in-process integration tests. A focused check of the actual orchestration workflow is:

```powershell
uv run pytest tests/integration/test_orchestrator.py -q
```

The evaluation suite exercises 14 scenarios, including normal planning, low budgets, invalid dates, unavailable agents, Redis failure, malformed model output and prompt injection. It gates deterministic metrics such as constraint satisfaction, source attribution and graceful degradation. Optional LLM judging is advisory.

### Infrastructure integration tests

> **Use disposable test services.** PostgreSQL fixtures drop/recreate tables and Redis fixtures flush their database. Their default ports overlap the demo stack. Do not run the unrestricted suite against stores you want to keep.

Run isolated containers on different ports:

```powershell
docker run --rm -d --name voyagemesh-test-postgres `
    -e POSTGRES_USER=voyagemesh -e POSTGRES_PASSWORD=voyagemesh `
    -e POSTGRES_DB=voyagemesh -p 127.0.0.1:55433:5432 pgvector/pgvector:pg16
docker run --rm -d --name voyagemesh-test-redis `
    -p 127.0.0.1:56380:6379 redis:7-alpine

$env:VM_TEST_DB_URL = "postgresql+asyncpg://voyagemesh:voyagemesh@localhost:55433/voyagemesh"
$env:VM_TEST_REDIS_URL = "redis://localhost:56380/0"
docker exec voyagemesh-test-postgres pg_isready -U voyagemesh
docker exec voyagemesh-test-redis redis-cli ping
```

After PostgreSQL accepts connections and Redis returns `PONG`:

```powershell
uv run pytest -q -m "not requires_groq and not requires_network" --cov --cov-report=term-missing
docker stop voyagemesh-test-postgres voyagemesh-test-redis
```

The `--rm` test containers are removed when stopped. Live Groq tests are opt-in through the `requires_groq` marker and need a valid runtime key file.

### Continuous integration

The [GitHub Actions workflow](.github/workflows/ci.yml) defines Python lint/type checks, tests with PostgreSQL/Redis and coverage, evaluation regression checks, frontend builds, security scans, Docker builds, Compose validation and a smoke test. The coverage floor is **80%**. Dependency and image vulnerability scans are advisory; those steps use `continue-on-error`.

Recorded validation was local. A successful hosted Actions run should be verified after publishing; no passing CI badge is claimed here.

## Observability

The `full` profile includes:

- **Jaeger:** traces across gateway, agent requests, tools and model calls where instrumented.
- **Prometheus:** request counts, latency, agent/tool activity, cache behaviour and model usage metrics.
- **Grafana:** eight dashboards covering system health, API/orchestration, agents/A2A, MCP/providers, LLM usage, cache/database, guardrails/security and product quality.
- **Structured logs:** request/trace context and sanitised failures.

The API returns request/trace headers for correlation. Agents propagate trace context and minimal user references; the user's access token is not forwarded to travel providers. The frontend shows request-level progress and result summaries, not a live stream of every LangGraph node.

<p align="center">
  <img src="images/image.png" alt="VoyageMesh Observability tab showing request ID, trip ID, generation time and links to Grafana, Prometheus and Jaeger" width="960" />
  <br />
  <em>Use the request ID to correlate a plan with service logs, then open dashboards, metrics or traces for more detail.</em>
</p>

## Repository layout

```text
Voyagemesh/
├── apps/
│   ├── frontend/              React and TypeScript UI
│   └── vm_api_gateway/        API, auth, persistence and service wiring
├── agents/
│   ├── vm_orchestrator/       LangGraph workflow, state and nodes
│   ├── vm_transport_agent/    Transport specialist
│   ├── vm_stay_agent/         Accommodation specialist
│   └── vm_itinerary_agent/    Itinerary specialist
├── mcp_servers/               Transport, lodging and destination tool servers
├── packages/                 Contracts, domain logic, auth, LLMs, cache,
│                             database, guardrails, harness and telemetry
├── data/                     Fixtures, RAG corpus and evaluation cases
├── evaluations/              Evaluation runner, metrics and regression gates
├── tests/                    Unit, integration, contract, security and evaluation
├── migrations/               Alembic database migrations
├── infra/                    Keycloak, OpenTelemetry, Prometheus and Grafana
├── docker/                   Dockerfiles and nginx configuration
├── scripts/                  Smoke test and Groq verification utilities
├── images/                   README visuals and application screenshots
├── .github/workflows/ci.yml  CI workflow
├── docker-compose.yml       Local service definitions and profiles
├── docker-compose.live.yml  Live providers and credential mounts
├── pyproject.toml            Python dependencies and tool configuration
└── uv.lock                   Locked Python dependency graph
```

## Validation and current limitations

During the **2026-09-23 README review**, **36 orchestration and API gateway tests passed**. The PowerShell examples parsed successfully, local documentation links resolved, and the documented Compose override passed configuration validation. These checks do not replace a live Docker run.

On **2026-10-01**, the images were rebuilt and the full Docker stack was started with `docker-compose.live.yml` and **no Duffel, LiteAPI or Geoapify key files**. The UI and gateway readiness checks returned HTTP 200; Keycloak login, a Berlin-to-Paris trip request through the frontend proxy and saved-trip retrieval succeeded. The request correctly returned `no_viable_plan`, missing-key warnings and an incomplete budget, without mock quotes. Open-Meteo resolved Paris and returned a **live weather forecast**. Application containers passed their health checks, and migration/seed jobs exited successfully.

That runtime check also uncovered and fixed an IPv6 mismatch in the frontend health probe and premature provider-client cleanup between stateless MCP requests. **64 provider and MCP contract tests passed**, including a regression through the actual MCP HTTP endpoint; lint and Python type checks passed. This verifies startup and missing-key behavior, not paid-provider account access or real flight/hotel quotes.

Current implementation boundaries include:

- **Travel inventory:** live Duffel flight, LiteAPI hotel and Geoapify place adapters are implemented; verification with real account credentials is pending. Omio rail/bus search is not integrated. Booking is not supported. Mock mode uses simulated inventory and fixtures.
- **RAG integration:** the database pipeline and MCP retrieval tool exist, but the itinerary workflow does not call that tool. The hashing embedder has limited semantic recall.
- **Agent registration:** addresses and routing are configured explicitly. Dynamic registration and protocol/schema negotiation are not implemented.
- **Audit persistence:** requests and plans are saved, but production callers do not populate the separate agent-execution and tool-audit tables.
- **Replanning:** incomplete budgets finalize without repeating unavailable searches. The live Compose override disables budget replans to avoid repeated metered queries; normal budget replanning remains available in the demo.
- **Output checks:** narrative screening is pattern based. The recommendation-membership check includes the recommendation itself, so it does not independently establish provider provenance; deterministic tool results and typed assembly provide the main grounding.
- **Cache and isolation:** live mode bypasses the plan cache. Mock-mode cache identity still uses budget bands; cached demo budgets need additional revalidation for different budgets in the same band. Stored trip lookup requires authentication but does not enforce per-user ownership.
- **Operational behaviour:** Redis-backed controls and database persistence fail open. Planning can succeed while rate limiting, deduplication or storage is unavailable. MCP tool-call counters accumulate for the server process lifetime, so extended demo use can require an MCP restart.
- **Deployment:** development credentials, internal unauthenticated MCP access and exposed infrastructure ports need hardening before shared/public deployment. Hosted CI execution remains to be verified.

The local and live Compose overrides above were used for the no-key Docker verification. This README follows the source implementation. Personal notes and assistant configuration stay local; only the root README is included as Markdown documentation in the public repository.

## Explore the implementation

| Topic | Source |
| --- | --- |
| Workflow and routing | [LangGraph graph](agents/vm_orchestrator/graph.py) and [planning nodes](agents/vm_orchestrator/nodes.py) |
| Agent discovery and delegation | [A2A client](packages/vm_harness/a2a_client.py) and [Agent Card contract](packages/vm_contracts/a2a.py) |
| Tools and provider adapters | [MCP servers](mcp_servers/) and [providers](packages/vm_domain/providers/) |
| Shared agent execution | [Agent harness](packages/vm_harness/) |
| Authentication and guardrails | [Auth](packages/vm_auth/), [guardrails](packages/vm_guardrails/) and [development realm](infra/keycloak/realm.json) |
| Persistence and retrieval | [Database package](packages/vm_database/) |
| API and interface | [Gateway](apps/vm_api_gateway/) and [frontend](apps/frontend/src/) |
| Containers and telemetry | [Compose](docker-compose.yml), [live override](docker-compose.live.yml) and [infrastructure](infra/) |
| Tests and CI | [Test suite](tests/), [evaluation runner](evaluations/) and [GitHub Actions](.github/workflows/ci.yml) |

## License

VoyageMesh is released under the [MIT License](LICENSE).
