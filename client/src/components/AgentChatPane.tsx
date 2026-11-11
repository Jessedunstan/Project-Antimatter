import React, { useState } from "react";
import { Send, Bot, Terminal, Wrench, CheckCircle, AlertTriangle, ChevronRight, ChevronDown } from "lucide-react";
import { api } from "../services/api";
import { AgentTrajectory, AgentStep } from "../types";

export const AgentChatPane: React.FC = () => {
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [trajectory, setTrajectory] = useState<AgentTrajectory | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});

  const handleRunAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim() || running) return;

    setRunning(true);
    setTrajectory({
      task,
      status: "thinking",
      steps: [],
    });

    try {
      const res = await api.runAgent(task);
      setTrajectory(res);
      // Auto-expand last step
      if (res.steps.length > 0) {
        setExpandedSteps((prev) => ({ ...prev, [res.steps.length]: true }));
      }
    } catch (err: any) {
      setTrajectory((prev) =>
        prev
          ? { ...prev, status: "error", error_message: err.message }
          : null
      );
    } finally {
      setRunning(false);
    }
  };

  const toggleStep = (stepNum: number) => {
    setExpandedSteps((prev) => ({ ...prev, [stepNum]: !prev[stepNum] }));
  };

  return (
    <div className="w-96 bg-[#18191e] border-l border-[#2b2e37] flex flex-col h-full select-none text-xs">
      {/* Pane Header */}
      <div className="h-9 bg-[#141518] border-b border-[#2b2e37] flex items-center justify-between px-3">
        <div className="flex items-center space-x-2 font-semibold text-white">
          <Bot size={15} className="text-blue-400" />
          <span>Antigravity ReAct Agent</span>
        </div>
        <div className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-[10px] text-emerald-400 uppercase font-mono">Ready</span>
        </div>
      </div>

      {/* Trajectory / Conversation Stream */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {!trajectory && (
          <div className="h-full flex flex-col items-center justify-center text-center text-[#9da3ae] p-4">
            <Bot size={32} className="text-[#3b4150] mb-2" />
            <p className="font-medium text-white mb-1">Autonomous Agent Idle</p>
            <p className="text-[11px] text-[#64748b]">
              Enter a task prompt below. The agent will inspect code, dispatch deterministic tools, and write patches.
            </p>
          </div>
        )}

        {trajectory && (
          <>
            {/* Task Prompt Banner */}
            <div className="bg-[#20222a] border border-[#2d313b] rounded p-2.5">
              <div className="text-[10px] uppercase font-bold text-[#64748b] mb-1">Goal</div>
              <div className="text-white font-medium">{trajectory.task}</div>
            </div>

            {/* ReAct Steps Trajectory */}
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
                        {/* Thought */}
                        {step.thought && (
                          <div className="text-[#9da3ae] leading-relaxed whitespace-pre-wrap">
                            {step.thought}
                          </div>
                        )}

                        {/* Tool Calls */}
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

            {/* Final Response or Error */}
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
          </>
        )}
      </div>

      {/* Task Input Prompt Bar */}
      <form onSubmit={handleRunAgent} className="p-3 border-t border-[#2b2e37] bg-[#141518]">
        <div className="relative">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Assign task to local agent..."
            rows={2}
            className="w-full bg-[#1e2026] border border-[#2b2e37] rounded p-2 text-white text-xs placeholder-[#64748b] focus:outline-none focus:border-blue-500 resize-none pr-8"
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
            className="absolute right-2 bottom-2.5 text-blue-400 hover:text-white disabled:opacity-40"
          >
            <Send size={14} className={running ? "animate-pulse" : ""} />
          </button>
        </div>
      </form>
    </div>
  );
};
