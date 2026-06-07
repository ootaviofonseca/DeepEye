# DeepEye Architecture

This document describes the end-to-end architecture of DeepEye as implemented in the codebase and as published in the SIGMOD 2026 paper.

> Source paper: *DeepEye: A Steerable Self-driving Data Agent System* (Li et al., SIGMOD Companion '26)
> arXiv: 2603.28889

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Four-Layer Architecture](#2-four-layer-architecture)
3. [Unified Node Protocol](#3-unified-node-protocol)
4. [Node Types](#4-node-types)
5. [Memory-Augmented Planner](#5-memory-augmented-planner)
6. [Agent Topology](#6-agent-topology)
7. [Workflow Engine](#7-workflow-engine)
8. [Sandbox System](#8-sandbox-system)
9. [Event System](#9-event-system)
10. [Infrastructure](#10-infrastructure)
11. [End-to-End Data Flow](#11-end-to-end-data-flow)
12. [Extension Guide](#12-extension-guide)

---

## 1. System Overview

DeepEye addresses three core limitations of existing "ChatBI" data agents:

| Challenge | Description |
|---|---|
| **C1 – Heterogeneity Gap** | Real decisions require joint analysis of SQL databases, CSV files, PDFs, and APIs together; existing systems are siloed (Text-to-SQL handles only databases, RAG handles only documents) |
| **C2 – Context Explosion** | Complex multi-step analysis in a single-agent context window causes hallucinations and instruction drift as history grows |
| **C3 – Unreliability & Inefficiency** | LLM-generated execution chains are opaque, hard to validate, and execute tasks sequentially missing parallelism opportunities |

DeepEye's three design responses:

- **Unified Multimodal Orchestration** – a standardized node protocol bridges heterogeneous data sources and output types under a single abstraction
- **Hierarchical Reasoning with Context Isolation** – complex intent is decomposed into context-isolated sub-agents (AgentNodes) and deterministic operators (ToolNodes); the global planner sees only their I/O interfaces
- **Database-Inspired Workflow Engine** – a four-phase pipeline (Compiler → Validator → Optimizer → Executor) guarantees structural correctness and enables runtime parallelism via topological scheduling

---

## 2. Four-Layer Architecture

The system is structured into four vertically integrated layers:

```
┌─────────────────────────────────────────────────────────┐
│  Data Sources                                           │
│  Structured (DB, CSV) │ Semi-structured │ Unstructured  │
│  (PostgreSQL/MySQL)   │ (JSON/XML)      │ (PDF/TXT/MD)  │
└──────────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│  Memory-Augmented Planner                               │
│  Short-term: Context Stack (dialogue, vars, feedback)   │
│  Long-term: Vector KB (schema metadata, SOPs, docs)     │
└──────────────────────────────┬──────────────────────────┘
                               │ Workflow DAG (JSON)
┌──────────────────────────────▼──────────────────────────┐
│  Workflow Engine                                        │
│  1. Compiler  2. Validator  3. Optimizer  4. Executor   │
└──────────────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────┐
│  Outputs                                                │
│  Data Videos │ Dashboards │ Analytical Reports          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Unified Node Protocol

Every capability in DeepEye is expressed as a node. The paper defines the formal protocol as a 5-tuple:

```
N = ⟨D, I, O, C, Φ⟩
```

| Symbol | Name | In Code |
|---|---|---|
| **D** | Semantic Description (natural language string used by the Planner for tool selection) | `NodeSpec.description` |
| **I** | Input Ports – set of `⟨Key, Type, Desc⟩` triplets | `NodeSpec.inputs: dict[str, Port]` |
| **O** | Output Ports – standardized artifacts produced | `NodeSpec.outputs: dict[str, Port]` |
| **C** | Configuration Parameters (static, e.g. model temperature) | `Node.params` + `NodeSpec.params_schema` |
| **Φ** | Internal Execution Logic | `BaseNode.build_handler()` |

### Port contract

Defined in [`packages/core/deepeye/workflows/models.py`](../packages/core/deepeye/workflows/models.py):

```python
class Port(BaseModel):
    schema_: str | dict | None   # type contract
    required: bool = False
    multiple: bool = False       # accepts multiple upstream edges
    default: Any | None = None
    description: str | None = None
```

### NodeSpec

Defined in [`packages/core/deepeye/workflows/registry.py`](../packages/core/deepeye/workflows/registry.py):

```python
class NodeSpec(BaseModel):
    type: str
    description: str | None
    params_schema: dict[str, Any] | None  # JSON Schema for params
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    version: str = "1.0"
```

`NodeSpec` is the **single source of truth**: it drives the Validator (pre-execution checks), the Executor (input resolution), the frontend node editor, and the Planner prompt (via the node catalog API at `/api/v1/workflow-nodes`).

### Node implementation contract

All nodes must subclass `BaseNode` ([`packages/backend/app/node/core/base.py`](../packages/backend/app/node/core/base.py)):

```python
class BaseNode:
    node_type: str          # e.g. "datasource.read"

    @classmethod
    def spec(cls) -> NodeSpec: ...          # required

    @classmethod
    def build_handler(cls, db, user_id, ...) -> NodeHandler | None: ...  # optional
```

---

## 4. Node Types

Based on how the execution logic Φ is implemented, nodes fall into two categories:

### 4.1 ToolNodes (Deterministic Operators)

Rule-based atomic operations. Execution is deterministic: `O = f_code(I, C)`. They guarantee idempotency and require no reasoning context.

### 4.2 AgentNodes (Probabilistic Reasoners)

LLM-driven sub-agents. Execution involves probabilistic reasoning: `O ~ P(· | I, C, W_local, T_internal)`. Each AgentNode runs with a **private context window** (`W_local`) isolated from the global planner. This context isolation prevents context explosion: the planner only sees the standardized I/O interfaces.

### 4.3 Registered Node Catalog

The full list of registered nodes is in [`packages/backend/app/node/__init__.py`](../packages/backend/app/node/__init__.py):

| Node Type | Module | Kind | Description |
|---|---|---|---|
| `datasource.read` | `app.node.data.datasource_read` | ToolNode | Read rows from a connected datasource (DB or file) |
| `sql.execute` | `app.node.data.sql_execute` | ToolNode | Execute a SQL query against a datasource |
| `rows.*` | `app.node.rows.basic` | ToolNode | Row-level operations (profile, filter, select, sort, join, aggregate) |
| `llm.answer` | `app.node.llm.answer` | AgentNode | Generate a natural language answer from upstream data |
| `python.code` | `app.node.code.python_code` | AgentNode | Generate and execute Python code in a sandboxed container |
| `dashboard.generate` | `app.node.dashboard.node` | AgentNode | NL2Dashboard: generate an interactive dashboard from data |
| `video.generator` | `app.node.video.node` | AgentNode | Produce an animated, narrated data video |
| `report.generate` | `app.node.report.node` | AgentNode | Write a structured analytical report |

New nodes are registered by adding a module to `NODE_MODULES` in `app/node/__init__.py`. `register_node_specs` and `register_node_handlers` auto-discover all `BaseNode` subclasses in each module.

---

## 5. Memory-Augmented Planner

The Planner translates a high-level user request R into an executable workflow DAG G. It uses a dual-memory architecture:

### Short-term Memory (Working Memory)

Maintains the **Context Stack**: multi-turn dialogue history, intermediate variable states, and runtime feedback from the Workflow Engine (e.g. SQL errors fed back for re-planning).

### Long-term Memory (Knowledge Base)

A vector database storing:
1. **Schema Metadata** – column names, table structures of connected datasources
2. **Domain Documentation** – user-uploaded PDFs, TXT, Markdown indexed for RAG
3. **SOP Experience** – verified historical workflows; successful runs are auto-archived and can be manually saved as expert templates

### Planning lifecycle

| Phase | Description |
|---|---|
| **Pre-Execution (RAG)** | Retrieves relevant SOPs and domain knowledge injected as in-context demonstrations |
| **Planning** | Reuses an existing SOP template or synthesizes a new DAG topology from the node catalog |
| **Runtime Feedback** | If execution fails (e.g. SQL syntax error), the error trace enters Working Memory and triggers re-planning |
| **Post-Execution** | Successfully executed workflows are auto-serialized to the Knowledge Base |

---

## 6. Agent Topology

The agent stack is built on LangGraph and follows a `Supervisor → WorkflowAgent → WorkflowEngine` chain:

```
User Request
      │
      ▼
SupervisorAgent          (packages/core/deepeye/agents/supervisor.py)
  │  Decides: direct-answer | ask-clarification | run-workflow
  │  Built on ReActAgent (LangGraph ReAct loop)
  │  System prompt injected dynamically with datasource context
  │
  ├─► WorkflowAgent       (packages/core/deepeye/agents/workflow_agent.py)
  │     Uses tools: create_workflow / update_workflow / run_workflow_from_file
  │     Generates a workflow JSON plan
  │
  └─► WorkflowEngine      (packages/core/deepeye/workflows/engine.py)
        Compiles, validates, optimizes, and executes the DAG
```

### SupervisorAgent

- Subclasses `ReActAgent` and runs the standard LangGraph `model → tool → model` loop
- System prompt is dynamically constructed per-request with the current session's datasource context
- Extracts `final_answer` from `WorkflowAgent` tool response to short-circuit the loop when the workflow is complete
- Located at: [`packages/core/deepeye/agents/supervisor.py`](../packages/core/deepeye/agents/supervisor.py)

### ReActAgent

- Base class for all agents in DeepEye
- Binds a `BaseChatModel` and a list of tools
- Supports optional `BaseCheckpointSaver` for persistent memory across turns
- Located at: [`packages/core/deepeye/agents/react_agent.py`](../packages/core/deepeye/agents/react_agent.py)

### WorkflowAgent

- Responsible for generating the workflow JSON (`WorkflowDraft`)
- Calls `create_workflow`, `update_workflow`, and `run_workflow_from_file` tools
- Located at: [`packages/core/deepeye/agents/workflow_agent.py`](../packages/core/deepeye/agents/workflow_agent.py)

### Task entry point

Agent execution is launched as a Celery task at [`packages/backend/app/tasks/agent_tasks.py`](../packages/backend/app/tasks/agent_tasks.py), which:
1. Instantiates the LLM model and callbacks
2. Injects available tools (workflow tools, sandbox bash tool)
3. Calls the Supervisor to execute

---

## 7. Workflow Engine

The Workflow Engine is the core of DeepEye's reliability guarantee. It transforms a logical workflow plan into verified, optimized execution through four phases (analogous to a DBMS query engine).

Code location: [`packages/core/deepeye/workflows/engine.py`](../packages/core/deepeye/workflows/engine.py)

### Phase 1 — Compilation

The Compiler (implemented in `WorkflowAgent`) parses the LLM-generated JSON plan into the typed domain model:

```
Workflow
  └── root: Graph
        ├── nodes: dict[str, Node]
        │     ├── id, type, params
        │     ├── inputs: dict[str, Port]
        │     └── outputs: dict[str, Port]
        └── edges: dict[str, Edge]
              ├── source: EdgeEndpoint (node_id, port_id)
              ├── target: EdgeEndpoint (node_id, port_id)
              ├── condition: str | None
              └── transform: str | None
```

Variable references between nodes (e.g. linking a `sql.execute` output port to a `dashboard.generate` input port) are resolved here.

### Phase 2 — Validation (Static Analysis)

`validate_workflow_graph()` in [`packages/core/deepeye/workflows/validation.py`](../packages/core/deepeye/workflows/validation.py) performs:

| Check | Error Code |
|---|---|
| Node type registered in `NodeRegistry` | `node.type.unknown` |
| Declared port exists in NodeSpec | `node.input.unknown`, `node.output.unknown` |
| Port schema compatibility between connected nodes | `edge.schema.incompatible` |
| Required inputs are connected | `input.required.missing` |
| Required params are present | `param.required.missing` |
| Non-multiple inputs don't have more than one incoming edge | `input.multiple.violation` |
| No cycles in the DAG | `graph.cycle` (DFS-based detection) |
| Group node internal graph consistency | `group.*` codes |

Any violation is reported before execution, preventing runtime failures.

### Phase 3 — Optimization

`_topological_sort()` in `engine.py` implements Kahn's Algorithm to produce a linear execution order respecting all data dependencies. Independent nodes (no shared dependency path) end up in the same conceptual execution layer and can be dispatched in parallel via the async Celery infrastructure.

Example: a `datasource.read` node fetching from a database and a `knowledge.search` node querying the vector store have no dependency relationship — the optimizer identifies them as parallel and dispatches both simultaneously.

### Phase 4 — Execution

`ExecutionEngine.run()` orchestrates the validated, sorted node sequence:

1. For each node in topological order:
   - Resolves inputs from upstream `NodeRun.outputs` (respecting edge `condition` and `transform` handlers)
   - Calls the registered `NodeHandler.execute(node, inputs, context)`
   - Validates handler outputs against the NodeSpec (no undeclared output ports allowed)
   - Records `NodeRun` (status, inputs, outputs, timestamps) in `ExecutionContext`
2. On any node failure: marks `context.status = "failed"`, returns partial context; error trace feeds back to the Planner for self-correction
3. On completion: `context.status = "success"` with all `NodeRun` records

The Executor dispatches work through Celery/Redis. The `RuntimeControl` service (port 8010) manages Docker container lifecycles for sandboxed node execution.

---

## 8. Sandbox System

ToolNodes and AgentNodes that execute user-derived code (Python, SQL) run inside isolated Docker containers. This prevents host filesystem access and limits network exposure.

```
SandboxManager                   (packages/backend/app/sandbox/manager.py)
  Session → DockerSandbox mapping
  Tracks activity, auto-stop, auto-destroy

DockerSandbox                    (packages/backend/app/sandbox/docker_sandbox.py)
  Container lifecycle (create, exec_command, stop, destroy)
  Mounts a workspace volume per session

ActivityTracker                  (packages/backend/app/sandbox/activity.py)
  Records last-used time; idle detection

RuntimeControl service           (packages/backend/app/runtime_control/main.py)
  Separate FastAPI process on port 8010
  Manages Docker daemon interactions
```

Sandbox lifecycle:
1. Agent requests `get_or_create_sandbox(session_id)`
2. Code execution happens via `exec_command()`
3. Cleanup task runs on a timer: idle > `SANDBOX_IDLE_TIMEOUT` → stop; idle > `SANDBOX_DESTROY_TIMEOUT` → destroy
4. On app shutdown, all sandboxes are destroyed

The sandbox image is defined in [`docker/Dockerfile.sandbox`](../docker/Dockerfile.sandbox). Dashboard rendering uses a separate image ([`docker/Dockerfile.dashboard`](../docker/Dockerfile.dashboard)) launched per-task.

---

## 9. Event System

DeepEye uses Server-Sent Events (SSE) to stream real-time execution state from the backend to the frontend.

```
AgentCallback                    (packages/backend/app/tasks/callbacks.py)
  Subscribes to LangChain callbacks (token, tool_start, tool_end)
  Writes events to Redis pub/sub channel keyed by session_id

MessageCollector                 (packages/backend/app/tasks/callbacks.py)
  Assembles step-level messages
  Persists completed turns to PostgreSQL

Frontend SSE listener
  Subscribes to /api/v1/chat/stream?session_id=...
  Renders live agent steps (thought, tool call, tool result, workflow events)
```

Event types pushed to the frontend include `tool_start`, `tool_end`, and `workflow_event` phases. The RFC in [`docs/rfcs/workflow_native_agent_refactor.md`](rfcs/workflow_native_agent_refactor.md) defines a target unified `workflow_event` protocol with the following phases:

```
turn_started → draft_created → validation_passed → run_started
  → node_started → node_log → node_finished → artifact_ready
  → run_finished → turn_finished
```

---

## 10. Infrastructure

DeepEye's cloud-native stack:

| Component | Technology | Role |
|---|---|---|
| API Server | FastAPI (port 8000) | REST endpoints, SSE streaming, auth |
| Async Worker | Celery + Redis | Long-running workflow and agent tasks |
| Runtime Control | FastAPI (port 8010) | Docker container management |
| Gateway | Nginx | Reverse proxy, routes `/api/*` to backend, `/` to frontend |
| Primary Database | PostgreSQL 15 | Users, sessions, workflows, artifacts, datasources |
| Cache & Broker | Redis 7 | Celery task queue, SSE pub/sub, session cache |
| Object Storage | MinIO (S3-compatible) | Workflow artifacts, generated videos, dashboard assets, uploaded files |
| Frontend | React 19 + Vite | Chat UI, workflow graph canvas, artifact preview panels |
| Sandbox | Docker containers | Isolated Python/code execution per session |

All services are composed in [`docker-compose.yml`](../docker-compose.yml). Database migrations are handled by Alembic and run automatically on stack startup.

### Network topology

```
Internet / Browser
      │
      ▼
   Nginx :8080
      │
      ├─► Backend API :8000  ──► PostgreSQL
      │         │              ─► Redis
      │         │              ─► MinIO
      │         │              ─► Runtime Control :8010
      │         │
      │         └─► Celery Worker (same image as backend)
      │
      └─► Frontend (static, served by Nginx)
```

---

## 11. End-to-End Data Flow

A complete request cycle for "Analyze the 2025 global sales performance":

```
1. User types message + selects @Sales Database @Financial Metrics in Chat UI
2. Frontend POST /api/v1/chat  {session_id, message, datasource_ids}
3. Backend creates ChatTurn, enqueues Celery task (agent_tasks.run_agent)
4. Celery worker picks up task:
   a. SupervisorAgent receives request + datasource context in system prompt
   b. Supervisor decides: run-workflow
   c. WorkflowAgent generates workflow JSON (DAG with nodes: datasource.read,
      knowledge.search, sql.execute, video.generator, dashboard.generate,
      report.generate)
5. WorkflowEngine.run(workflow):
   a. Compiler: parses JSON → Workflow / Graph / Node objects
   b. Validator: checks all node types, port schemas, required params, DAG acyclicity
   c. Optimizer: topological sort; datasource.read and knowledge.search have no
      dependency → same execution layer → dispatched in parallel
   d. Executor: runs each layer; outputs of earlier nodes resolve inputs of later nodes
6. Each node handler executes (some in Docker sandbox):
   - datasource.read → fetches rows from PostgreSQL sales DB
   - knowledge.search → queries MinIO-backed vector store for financial docs
   - sql.execute → runs generated SQL, returns aggregated rows
   - video.generator → produces .mp4 artifact stored in MinIO
   - dashboard.generate → renders dashboard, deploys to container
   - report.generate → writes Markdown report stored in MinIO
7. ExecutionContext with all NodeRun records is serialized
8. Artifacts stored in MinIO, metadata in PostgreSQL
9. AgentCallback emits workflow_event phases to Redis → SSE → frontend
10. Frontend renders workflow DAG, artifact panels (video player, dashboard iframe,
    report viewer)
```

---

## 12. Extension Guide

### Adding a new Node

1. Create a `BaseNode` subclass in `packages/backend/app/node/<domain>/`:

```python
from deepeye.workflows.registry import NodeSpec
from deepeye.workflows.models import Port
from app.node.core.base import BaseNode
from deepeye.workflows.engine import NodeHandler

class MyExportNode(BaseNode):
    node_type = "data.export_csv"

    @classmethod
    def spec(cls) -> NodeSpec:
        return NodeSpec(
            type=cls.node_type,
            description="Export rows to a CSV file",
            inputs={
                "rows": Port(schema_="list[dict]", required=True, multiple=True),
            },
            outputs={
                "path": Port(schema_="string"),
            },
            params_schema={
                "filename": {"type": "string", "required": True},
            },
        )

    @classmethod
    def build_handler(cls, db, user_id, sandbox=None) -> NodeHandler:
        async def handler(node, inputs, context):
            rows = inputs.get("rows", [])
            filename = node.params.get("filename", "export.csv")
            # ... write to sandbox workspace ...
            return {"path": f"/workspace/{filename}"}
        return type("H", (), {"execute": staticmethod(handler)})()
```

2. Add the module path to `NODE_MODULES` in [`packages/backend/app/node/__init__.py`](../packages/backend/app/node/__init__.py).

3. The node is now automatically available via `/api/v1/workflow-nodes` and usable by the Planner.

### Adding a new Agent

1. Create a subclass of `ReActAgent` in `packages/core/deepeye/agents/`:

```python
from deepeye.agents.react_agent import ReActAgent

class LogAnalysisAgent(ReActAgent):
    def __init__(self, model, tools=None, **kwargs):
        super().__init__(
            model=model,
            tools=tools or [],
            system_prompt="You are a log analysis assistant.",
            **kwargs,
        )
```

2. Create the tool that wraps it (as a LangChain tool) and add it to the Supervisor's tool list in `packages/backend/app/tasks/agent_tasks.py`.

3. Update the Supervisor system prompt to include the decision rule for when to invoke the new agent.

---

## Related Documents

- [Workflow node system (detail)](workflow_node_system.md) — node registration, BaseNode contract, extension examples
- [Agent module (detail)](agent_module.md) — agent call chain, callbacks, extension examples
- [Sandbox system (detail)](sandbox_system.md) — Docker sandbox lifecycle, cleanup, extension
- [RFC: Workflow-native agent refactor](rfcs/workflow_native_agent_refactor.md) — target architecture, domain model (Turn/Draft/Run/Artifact), unified event protocol, migration plan
- [RFC: Artifact protocol](rfcs/artifact_protocol.md) — artifact interchange format
- [Security model](security_model.md) — sandbox boundaries, auth, deployment hardening