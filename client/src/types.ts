export type FileType = "file" | "directory";

export interface WorkspaceEntry {
  name: string;
  type: FileType;
  rel_path: string;
}

export interface Diagnostic {
  file_path: string;
  message: string;
  severity: number; // 1: Error, 2: Warning
  start_line: number;
  start_char: number;
  end_line: number;
  end_char: number;
}

export interface SubagentTelemetry {
  id: string;
  name: string;
  role: string;
  status: string;
  workspace_mode: string;
  has_error?: boolean;
}

export interface ArtifactMeta {
  id: string;
  title: string;
  summary: string;
  artifact_type: "markdown" | "html" | "mermaid" | "diff" | "svg";
  version: number;
  user_facing: boolean;
}

export interface CommandResult {
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
  timed_out: boolean;
}

export interface ToolCallPayload {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

export interface AgentStep {
  step_number: number;
  thought?: string;
  tool_calls: ToolCallPayload[];
  tool_results: Record<string, any>;
  timestamp: number;
}

export interface AgentTrajectory {
  task: string;
  status: "idle" | "thinking" | "executing_tool" | "completed" | "error";
  steps: AgentStep[];
  final_response?: string;
  error_message?: string;
}

export interface SystemHealth {
  status: string;
  app_name: string;
  version: string;
  profile: string;
  backend: string;
  offline: boolean;
  llm: {
    status: string;
    backend: string;
    models?: string[];
  };
}
