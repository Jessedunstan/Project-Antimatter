import {
  ArtifactMeta,
  CommandResult,
  SystemHealth,
  SubagentTelemetry,
  WorkspaceEntry,
  AgentTrajectory,
  SwarmTrajectory,
} from "../types";

const BASE_URL = "http://127.0.0.1:8000";

export const api = {
  async getHealth(): Promise<SystemHealth> {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async listFiles(subpath = ""): Promise<WorkspaceEntry[]> {
    const res = await fetch(`${BASE_URL}/files?subpath=${encodeURIComponent(subpath)}`);
    if (!res.ok) throw new Error("Failed to list files");
    const data = await res.json();
    return data.entries;
  },

  async readFile(path: string): Promise<string> {
    const res = await fetch(`${BASE_URL}/file/read?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error(`Failed to read file: ${path}`);
    const data = await res.json();
    return data.content;
  },

  async writeFile(path: string, content: string): Promise<number> {
    const res = await fetch(`${BASE_URL}/file/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, content }),
    });
    if (!res.ok) throw new Error("Failed to write file");
    const data = await res.json();
    return data.bytes_written;
  },

  async runTerminal(command: string): Promise<CommandResult> {
    const res = await fetch(`${BASE_URL}/terminal/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
    if (!res.ok) throw new Error("Terminal execution failed");
    return res.json();
  },

  async getAutocomplete(prefix: string, suffix: string, filePath?: string): Promise<string> {
    try {
      const res = await fetch(`${BASE_URL}/autocomplete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefix, suffix, file_path: filePath }),
      });
      if (!res.ok) return "";
      const data = await res.json();
      return data.completion || "";
    } catch {
      return "";
    }
  },

  async runAgent(task: string, maxSteps = 15): Promise<AgentTrajectory> {
    const res = await fetch(`${BASE_URL}/agent/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, max_steps: maxSteps }),
    });
    if (!res.ok) throw new Error("Agent run failed");
    return res.json();
  },

  async runSwarm(params: {
    goal: string;
    target_file: string;
    test_file?: string;
    test_command?: string;
  }): Promise<SwarmTrajectory> {
    const res = await fetch(`${BASE_URL}/swarm/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error("Swarm run failed");
    return res.json();
  },

  async listSubagents(): Promise<SubagentTelemetry[]> {
    const res = await fetch(`${BASE_URL}/subagents`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.subagents;
  },

  async listArtifacts(): Promise<ArtifactMeta[]> {
    const res = await fetch(`${BASE_URL}/artifacts`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.artifacts;
  },

  getArtifactHtmlUrl(id: string): string {
    return `${BASE_URL}/artifacts/${id}/html`;
  },
};
