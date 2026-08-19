# Agentic AI — Multi-Agent Essay & Blog Writer

Agentic AI refers to autonomous systems where one or more AI agents work together to achieve a complex goal through **planning, reasoning, tool use, and collaboration**.

---

## 🤖 Core Concepts: Agent vs. Agentic System

| Concept | What It Is | Role |
| :--- | :--- | :--- |
| **AI Agent** | A single, task-focused worker | Performs one specific job (e.g., searches the web, writes one section, calls a tool). |
| **Agentic AI (Multi-Agent System)** | The orchestrator & team | Coordinates multiple specialized agents to solve a complete end-to-end goal. |

---

## 🏗️ Essay & Blog Writer Architecture

This project uses a **Map-Reduce / Multi-Agent workflow** to plan, research, write, and format full-length essays and blogs.

```
                      User Prompt / Topic
                              │
                              ▼
                       ┌──────────────┐
                       │ Router Node  │
                       └──────┬───────┘
                              │
             Needs Web Research?
            ├─── YES ──▶ [Research Agent (Tavily API)] ──┐
            └─── NO  ───────────────────────────────────┼──▶ ┌──────────────┐
                                                             │ Planner Node │
                                                             └──────┬───────┘
                                                                    │
                                                 Creates Section-wise Outline
                                                                    │
                             ┌──────────────────────────────────────┼──────────────────────────────────────┐
                             ▼                                      ▼                                      ▼
                   ┌───────────────────┐                  ┌───────────────────┐                  ┌───────────────────┐
                   │  Worker Agent 1   │                  │  Worker Agent 2   │                  │  Worker Agent N   │
                   │ (Writes Section 1)│                  │ (Writes Section 2)│                  │ (Writes Section N)│
                   └─────────┬─────────┘                  └─────────┬─────────┘                  └─────────┬─────────┘
                             │                                      │                                      │
                             └──────────────────────────────────────┼──────────────────────────────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │ Reducer Node │
                                                            └──────┬───────┘
                                                                   │
                                                • Combines all written sections
                                                • Identifies image placement spots
                                                • Generates & embeds visuals
                                                                   │
                                                                   ▼
                                                          Final Complete Blog
```

---

## 🔍 Node Breakdown

1. **Router Node:** Decides if the requested topic requires real-time web search or can be answered with internal model knowledge.
2. **Research Node (Tavily):** Performs web searches to gather up-to-date facts, statistics, and citations.
3. **Planner Node:** Breaks the topic into structured sections using a strict schema (`Plan` with a list of `Task` objects).
4. **Worker Nodes (Map):** Multiple agents run in parallel, each responsible for writing one section with depth.
5. **Reducer Node (Reduce):**
   - Merges all generated sections into a cohesive piece.
   - Determines where illustrations/images are needed.
   - Generates and embeds images in the final markdown draft.

---

## 🚀 Development Roadmap

- [x] **Phase 1: Basic Text Agent**
  - Planner + Parallel Section Writers + Reducer (Text-only blog).
- [ ] **Phase 2: Live Research Integration**
  - Add Tavily web search tool to the research node.
- [ ] **Phase 3: Visual Generation**
  - Add image generation & automatic image placement logic in Reducer.
- [ ] **Phase 4: User Interface (GUI)**
  - Build an interactive Web UI (Streamlit / Gradio / Next.js) for topic input and blog preview.
