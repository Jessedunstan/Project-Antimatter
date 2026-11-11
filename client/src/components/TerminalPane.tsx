import React, { useState } from "react";
import { Terminal as TerminalIcon, Play, CheckCircle2, AlertCircle, Clock } from "lucide-react";
import { api } from "../services/api";
import { CommandResult } from "../types";

export const TerminalPane: React.FC = () => {
  const [cmd, setCmd] = useState("");
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<Array<{ command: string; result: CommandResult }>>([]);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cmd.trim() || running) return;

    const commandToRun = cmd;
    setCmd("");
    setRunning(true);

    try {
      const res = await api.runTerminal(commandToRun);
      setHistory((prev) => [...prev, { command: commandToRun, result: res }]);
    } catch (err: any) {
      setHistory((prev) => [
        ...prev,
        {
          command: commandToRun,
          result: {
            exit_code: -1,
            stdout: "",
            stderr: err.message,
            duration_ms: 0,
            timed_out: false,
          },
        },
      ]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#121316] text-xs font-mono">
      {/* Header */}
      <div className="h-9 bg-[#141518] border-b border-[#2b2e37] flex items-center justify-between px-3 text-[#9da3ae]">
        <div className="flex items-center space-x-2 text-white font-sans font-semibold">
          <TerminalIcon size={14} className="text-emerald-400" />
          <span>Sovereign Sandboxed Terminal (PowerShell / Bash)</span>
        </div>
        <button
          onClick={() => setHistory([])}
          className="text-[10px] text-[#64748b] hover:text-white"
        >
          Clear
        </button>
      </div>

      {/* Terminal History */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {history.map((item, idx) => (
          <div key={idx} className="space-y-1">
            <div className="flex items-center text-blue-400">
              <span className="text-emerald-400 mr-2">$</span>
              <span className="font-bold">{item.command}</span>
            </div>
            {item.result.stdout && (
              <pre className="text-[#e2e8f0] whitespace-pre-wrap pl-4 border-l border-[#282b35] text-[11px]">
                {item.result.stdout}
              </pre>
            )}
            {item.result.stderr && (
              <pre className="text-red-400 whitespace-pre-wrap pl-4 border-l border-red-500/30 text-[11px]">
                {item.result.stderr}
              </pre>
            )}
            <div className="flex items-center space-x-3 text-[10px] text-[#64748b] pl-4 pt-1">
              <span className="flex items-center">
                {item.result.exit_code === 0 ? (
                  <CheckCircle2 size={11} className="mr-1 text-emerald-400" />
                ) : (
                  <AlertCircle size={11} className="mr-1 text-red-400" />
                )}
                Exit {item.result.exit_code}
              </span>
              <span className="flex items-center">
                <Clock size={10} className="mr-1" />
                {item.result.duration_ms.toFixed(1)}ms
              </span>
            </div>
          </div>
        ))}
        {history.length === 0 && (
          <div className="text-[#64748b] italic">Terminal ready. Enter commands to execute in workspace sandbox.</div>
        )}
      </div>

      {/* Command Input Prompt */}
      <form onSubmit={handleRun} className="p-2 border-t border-[#2b2e37] bg-[#141518] flex items-center space-x-2">
        <span className="text-emerald-400 font-bold">$</span>
        <input
          type="text"
          value={cmd}
          onChange={(e) => setCmd(e.target.value)}
          placeholder="pytest, git status, python script.py..."
          className="flex-1 bg-transparent text-white focus:outline-none text-xs font-mono"
        />
        <button
          type="submit"
          disabled={running || !cmd.trim()}
          className="px-2 py-1 bg-emerald-700 hover:bg-emerald-600 text-white rounded text-[11px] flex items-center disabled:opacity-40"
        >
          <Play size={10} className="mr-1" /> Run
        </button>
      </form>
    </div>
  );
};
