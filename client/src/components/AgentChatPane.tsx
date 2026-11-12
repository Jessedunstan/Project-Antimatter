import React, { useState } from "react";
import {
  Send,
  Bot,
  Terminal,
  Wrench,
  CheckCircle,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
  Layers,
  Search,
  Compass,
  Cpu,
  ShieldCheck,
  RotateCcw,
} from "lucide-react";
import { api } from "../services/api";
import { AgentTrajectory, SwarmTrajectory, SwarmStep } from "../types";

export const AgentChatPane: React.FC = () => {
  const [mode, setMode] = useState<"react" | "swarm">("swarm");
  const [task, setTask] = useState("");
  const [targetFile, setTargetFile] = useState("antimatter/core/config.py");
  const [testCommand, setTestCommand] = useState("");
  const [running, setRunning] = useState(false);

  // ReAct Trajectory State
  const [trajectory, setTrajectory] = useState<AgentTrajectory | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});

  // Swarm Trajectory State
  const [swarmTrajectory, setSwarmTrajectory] = useState<SwarmTrajectory | null>(null);
  const [expandedStages, setExpandedStages] = useState<Record<string, boolean>>({
    scout: true,
    architect: true,
    coder: true,
    verifier: true,
  });

  const handleRunAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim() || running) return;

    setRunning(true);

    if (mode === "react") {
      setTrajectory({
        task,
        status: "thinking",
        steps: [],
      });

      try {
        const res = await api.runAgent(task);
        setTrajectory(res);
        if (res.steps.length > 0) {
          setExpandedSteps((prev) => ({ ...prev, [res.steps.length]: true }));
        }
      } catch (err: any) {
        setTrajectory((prev) =>
          prev ? { ...prev, status: "error", error_message: err.message } : null
        );
      } finally {
        setRunning(false);
      }
    } else {
      // Swarm Mode
      setSwarmTrajectory({
        task_id: "init",
        user_goal: task,
        overall_status: "running",
        stages: [
          { stage: "scout", status: "running", summary: "Scanning codebase symbols...", duration_sec: 0 },
          { stage: "architect", status: "pending", summary: "Awaiting blueprint formulation...", duration_sec: 0 },
          { stage: "coder", status: "pending", summary: "Awaiting patch execution...", duration_sec: 0 },
          { stage: "verifier", status: "pending", summary: "Awaiting test execution...", duration_sec: 0 },
        ],
        total_duration_sec: 0,
        final_report: "",
        retries_count: 0,
      });

      try {
        const res = await api.runSwarm({
          goal: task,
          target_file: targetFile,
          test_command: testCommand || undefined,
        });
        setSwarmTrajectory(res);
      } catch (err: any) {
        setSwarmTrajectory((prev) =>
          prev
            ? {
                ...prev,
                overall_status: "failed",
                final_report: `Swarm Execution Error: ${err.message}`,
              }
            : null
        );
      } finally {
        setRunning(false);
      }
    }
  };

  const toggleStep = (stepNum: number) => {
    setExpandedSteps((prev) => ({ ...prev, [stepNum]: !prev[stepNum] }));
  };

  const toggleStage = (stageName: string) => {
    setExpandedStages((prev) => ({ ...prev, [stageName]: !prev[stageName] }));
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case "scout":
        return <Search size={14} className="text-cyan-400" />;
      case "architect":
        return <Compass size={14} className="text-purple-400" />;
      case "coder":
        return <Cpu size={14} className="text-amber-400" />;
      case "verifier":
        return <ShieldCheck size={14} className="text-emerald-400" />;
      default:
        return <Bot size={14} className="text-blue-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "passed":
        return <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono">PASSED</span>;
      case "failed":
        return <span className="text-[10px] bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded font-mono">FAILED</span>;
      case "running":
        return <span className="text-[10px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded font-mono animate-pulse">RUNNING</span>;
      default:
        return <span className="text-[10px] bg-gray-500/20 text-gray-400 px-1.5 py-0.5 rounded font-mono">PENDING</span>;
    }
  };

  return (
    <div className="w-96 bg-[#18191e] border-l border-[#2b2e37] flex flex-col h-full select-none text-xs">
      {/* Pane Header */}
      <div className="h-10 bg-[#141518] border-b border-[#2b2e37] flex items-center justify-between px-3">
        <div className="flex items-center space-x-2 font-semibold text-white">
          <Layers size={15} className="text-indigo-400" />
          <span>Antimatter Agent</span>
        </div>
        {/* Mode Selector */}
        <div className="flex items-center bg-[#1e2026] p-0.5 rounded border border-[#2d313b]">
          <button
            onClick={() => setMode("swarm")}
            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
              mode === "swarm" ? "bg-indigo-600 text-white" : "text-[#9da3ae] hover:text-white"
            }`}
          >
            Swarm
          </button>
          <button
            onClick={() => setMode("react")}
            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
              mode === "react" ? "bg-blue-600 text-white" : "text-[#9da3ae] hover:text-white"
            }`}
          >
            ReAct
          </button>
        </div>
      </div>

      {/* Trajectory / Stream Area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {mode === "swarm" ? (
          <>
            {!swarmTrajectory && (
              <div className="h-full flex flex-col items-center justify-center text-center text-[#9da3ae] p-4">
                <Layers size={32} className="text-[#3b4150] mb-2" />
                <p className="font-medium text-white mb-1">Autonomous Swarm Engine</p>
                <p className="text-[11px] text-[#64748b]">
                  Specialists (Scout 🔍, Architect 📐, Coder 💻, Verifier 🧪) decompose, execute, and self-heal patches collaboratively.
                </p>
              </div>
            )}

            {swarmTrajectory && (
              <div className="space-y-3">
                {/* Goal Banner */}
                <div className="bg-[#20222a] border border-[#2d313b] rounded p-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] uppercase font-bold text-[#64748b]">Swarm Goal</span>
                    {getStatusBadge(swarmTrajectory.overall_status)}
                  </div>
                  <div className="text-white font-medium">{swarmTrajectory.user_goal}</div>
                  {swarmTrajectory.retries_count > 0 && (
                    <div className="mt-1 text-[10px] text-amber-400 flex items-center">
                      <RotateCcw size={10} className="mr-1" />
                      Self-healed across {swarmTrajectory.retries_count} retries
                    </div>
                  )}
                </div>

                {/* 4-Stage Specialist Pipeline */}
                <div className="space-y-2">
                  {swarmTrajectory.stages.map((stg) => {
                    const isExpanded = !!expandedStages[stg.stage];
                    return (
                      <div
                        key={stg.stage}
                        className="border border-[#2d313b] bg-[#1a1c22] rounded overflow-hidden"
                      >
                        <div
                          onClick={() => toggleStage(stg.stage)}
                          className="px-2.5 py-1.5 bg-[#20222a] flex items-center justify-between cursor-pointer hover:bg-[#252832]"
                        >
                          <div className="flex items-center space-x-1.5 font-medium text-white capitalize">
                            {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            {getStageIcon(stg.stage)}
                            <span>{stg.stage} Specialist</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            {stg.duration_sec > 0 && (
                              <span className="text-[10px] text-[#64748b]">{stg.duration_sec.toFixed(1)}s</span>
                            )}
                            {getStatusBadge(stg.status)}
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="p-2.5 space-y-2 border-t border-[#2d313b] text-[11px]">
                            <div className="text-[#c2c7d0] leading-relaxed">{stg.summary}</div>
                            {stg.evidence && Object.keys(stg.evidence).length > 0 && (
                              <div className="bg-[#121316] border border-[#282b35] rounded p-2 font-mono text-[10px] text-[#9da3ae] max-h-36 overflow-y-auto whitespace-pre-wrap">
                                {JSON.stringify(stg.evidence, null, 2)}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Final Report */}
                {swarmTrajectory.final_report && (
                  <div
                    className={`rounded p-3 border ${
                      swarmTrajectory.overall_status === "passed"
                        ? "bg-emerald-950/20 border-emerald-500/30 text-emerald-200"
                        : "bg-red-950/20 border-red-500/30 text-red-200"
                    }`}
                  >
                    <div className="flex items-center font-bold mb-1">
                      {swarmTrajectory.overall_status === "passed" ? (
                        <CheckCircle size={13} className="mr-1.5 text-emerald-400" />
                      ) : (
                        <AlertTriangle size={13} className="mr-1.5 text-red-400" />
                      )}
                      Swarm Result
                    </div>
                    <div className="whitespace-pre-wrap text-[11px] leading-relaxed font-mono">
                      {swarmTrajectory.final_report}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          /* ReAct Mode */
          <>
            {!trajectory && (
              <div className="h-full flex flex-col items-center justify-center text-center text-[#9da3ae] p-4">
                <Bot size={32} className="text-[#3b4150] mb-2" />
                <p className="font-medium text-white mb-1">Autonomous ReAct Loop</p>
                <p className="text-[11px] text-[#64748b]">
                  Single-agent tool caller executing cyclical Thought-Action-Observation turns.
                </p>
              </div>
            )}

            {trajectory && (
              <div className="space-y-3">
                <div className="bg-[#20222a] border border-[#2d313b] rounded p-2.5">
                  <div className="text-[10px] uppercase font-bold text-[#64748b] mb-1">Goal</div>
                  <div className="text-white font-medium">{trajectory.task}</div>
                </div>

                <div className="space-y-2">
                  {trajectory.steps.map((step) => {
                    const isExpanded = !!expandedSteps[step.step_number];
                    return (
                      <div
                        key={step.step_number}
                        className="border border-[#2d313b] bg-[#1a1c22] rounded overflow-hidden"
                      >
                        <div
                          onClick={() => toggleStep(step.step_number)}
                          className="px-2.5 py-1.5 bg-[#20222a] flex items-center justify-between cursor-pointer hover:bg-[#252832]"
                        >
                          <div className="flex items-center space-x-1.5 font-medium text-white">
                            {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            <span>Step {step.step_number}</span>
                            {step.tool_calls.length > 0 && (
                              <span className="text-[9px] px-1 bg-blue-500/20 text-blue-300 rounded font-mono">
                                {step.tool_calls.map((t) => t.name).join(", ")}
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] text-[#64748b]">ReAct</span>
                        </div>

                        {isExpanded && (
                          <div className="p-2.5 space-y-2 border-t border-[#2d313b] text-[11px]">
                            {step.thought && (
                              <div className="text-[#9da3ae] leading-relaxed whitespace-pre-wrap">
                                {step.thought}
                              </div>
                            )}

                            {step.tool_calls.map((tc) => (
                              <div
                                key={tc.id}
                                className="bg-[#121316] border border-[#282b35] rounded p-2 font-mono text-[10px]"
                              >
                                <div className="flex items-center text-blue-400 font-bold mb-1">
                                  <Wrench size={11} className="mr-1" />
                                  {tc.name}
                                </div>
                                <div className="text-[#a6acb8] overflow-x-auto">
                                  {JSON.stringify(tc.arguments, null, 2)}
                                </div>
                                {step.tool_results[tc.id] && (
                                  <div className="mt-1.5 pt-1.5 border-t border-[#22252e] text-emerald-400 max-h-24 overflow-y-auto whitespace-pre-wrap">
                                    {step.tool_results[tc.id]}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {trajectory.status === "completed" && trajectory.final_response && (
                  <div className="bg-emerald-950/20 border border-emerald-500/30 rounded p-3 text-emerald-200">
                    <div className="flex items-center font-bold text-emerald-400 mb-1">
                      <CheckCircle size={13} className="mr-1.5" /> Task Finished
                    </div>
                    <div className="whitespace-pre-wrap text-[11px] leading-relaxed">
                      {trajectory.final_response}
                    </div>
                  </div>
                )}

                {trajectory.status === "error" && (
                  <div className="bg-red-950/20 border border-red-500/30 rounded p-3 text-red-200">
                    <div className="flex items-center font-bold text-red-400 mb-1">
                      <AlertTriangle size={13} className="mr-1.5" /> Execution Error
                    </div>
                    <div className="text-[11px]">{trajectory.error_message}</div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Task Input Prompt Bar */}
      <form onSubmit={handleRunAgent} className="p-3 border-t border-[#2b2e37] bg-[#141518] space-y-2">
        {mode === "swarm" && (
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={targetFile}
              onChange={(e) => setTargetFile(e.target.value)}
              placeholder="Target file (e.g. antimatter/core/config.py)"
              className="flex-1 bg-[#1e2026] border border-[#2b2e37] rounded px-2 py-1 text-white text-[11px] placeholder-[#64748b] focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        )}

        <div className="relative">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder={
              mode === "swarm"
                ? "Assign goal to Autonomous Swarm (e.g. Add validation to config)..."
                : "Assign task to local ReAct agent..."
            }
            rows={2}
            className="w-full bg-[#1e2026] border border-[#2b2e37] rounded p-2 text-white text-xs placeholder-[#64748b] focus:outline-none focus:border-indigo-500 resize-none pr-8"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleRunAgent(e);
              }
            }}
          />
          <button
            type="submit"
            disabled={running || !task.trim()}
            className={`absolute right-2 bottom-2.5 ${
              mode === "swarm" ? "text-indigo-400" : "text-blue-400"
            } hover:text-white disabled:opacity-40`}
          >
            <Send size={14} className={running ? "animate-pulse" : ""} />
          </button>
        </div>
      </form>
    </div>
  );
};
