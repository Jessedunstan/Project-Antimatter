import React, { useEffect, useState } from "react";
import {
  Folder,
  FileCode,
  Cpu,
  Bot,
  RefreshCw,
  HardDrive,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { api } from "../services/api";
import { WorkspaceEntry, SubagentTelemetry, SystemHealth } from "../types";

interface SidebarProps {
  currentFile: string;
  onSelectFile: (path: string) => void;
  activeView: string;
  onSelectView: (view: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentFile,
  onSelectFile,
  activeView,
  onSelectView,
}) => {
  const [files, setFiles] = useState<WorkspaceEntry[]>([]);
  const [subagents, setSubagents] = useState<SubagentTelemetry[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshData = async () => {
    setLoading(true);
    try {
      const [f, s, h] = await Promise.all([
        api.listFiles(),
        api.listSubagents(),
        api.getHealth(),
      ]);
      setFiles(f);
      setSubagents(s);
      setHealth(h);
    } catch (e) {
      console.warn("Failed fetching sidebar data", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-64 bg-[#18191e] border-r border-[#2b2e37] flex flex-col h-full select-none text-xs">
      {/* App Header */}
      <div className="p-3 border-b border-[#2b2e37] flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />
          <span className="font-bold text-sm tracking-wide text-white">ANTIMATTER</span>
        </div>
        <button
          onClick={refreshData}
          className="text-[#9da3ae] hover:text-white p-1 rounded hover:bg-[#252830]"
          title="Refresh Workspace"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Navigation Views */}
      <div className="flex border-b border-[#2b2e37] bg-[#141518]">
        {["editor", "canvas", "terminal"].map((view) => (
          <button
            key={view}
            onClick={() => onSelectView(view)}
            className={`flex-1 py-2 text-center uppercase tracking-wider font-semibold ${
              activeView === view
                ? "text-blue-400 border-b-2 border-blue-500 bg-[#1e2026]"
                : "text-[#9da3ae] hover:text-white"
            }`}
          >
            {view}
          </button>
        ))}
      </div>

      {/* Workspace File Explorer */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="text-[10px] font-bold text-[#64748b] uppercase tracking-wider px-2 py-1 flex items-center">
          <Folder size={12} className="mr-1.5" /> Workspace Explorer
        </div>
        <div className="space-y-0.5 mt-1">
          {files.map((file) => (
            <div
              key={file.rel_path}
              onClick={() => onSelectFile(file.rel_path)}
              className={`flex items-center px-2 py-1.5 rounded cursor-pointer transition-colors ${
                currentFile === file.rel_path
                  ? "bg-blue-600/20 text-blue-400 font-medium"
                  : "text-[#c2c7d0] hover:bg-[#22242c] hover:text-white"
              }`}
            >
              <FileCode size={13} className="mr-2 text-blue-400" />
              <span className="truncate">{file.name}</span>
            </div>
          ))}
          {files.length === 0 && (
            <div className="text-[#64748b] italic px-2 py-2">Workspace empty</div>
          )}
        </div>

        {/* Active Subagents Section */}
        <div className="mt-6">
          <div className="text-[10px] font-bold text-[#64748b] uppercase tracking-wider px-2 py-1 flex items-center">
            <Bot size={12} className="mr-1.5" /> Active Subagents ({subagents.length})
          </div>
          <div className="space-y-1 mt-1">
            {subagents.map((sub) => (
              <div
                key={sub.id}
                className="bg-[#20222a] border border-[#2d313b] rounded p-2 text-[11px]"
              >
                <div className="flex items-center justify-between font-semibold text-white">
                  <span>{sub.name}</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 uppercase">
                    {sub.workspace_mode}
                  </span>
                </div>
                <div className="text-[#9da3ae] text-[10px] truncate mt-0.5">{sub.role}</div>
                <div className="flex items-center mt-1 text-[10px] text-emerald-400">
                  <CheckCircle2 size={10} className="mr-1" /> {sub.status}
                </div>
              </div>
            ))}
            {subagents.length === 0 && (
              <div className="text-[#64748b] italic px-2 py-1 text-[11px]">No active subagents</div>
            )}
          </div>
        </div>
      </div>

      {/* System & Hardware Health Footer */}
      <div className="p-3 border-t border-[#2b2e37] bg-[#141518] text-[11px] space-y-1.5">
        <div className="flex items-center justify-between text-[#9da3ae]">
          <span className="flex items-center">
            <Cpu size={12} className="mr-1 text-purple-400" /> Backend
          </span>
          <span className="font-mono text-white">{health?.backend || "ollama"}</span>
        </div>
        <div className="flex items-center justify-between text-[#9da3ae]">
          <span className="flex items-center">
            <HardDrive size={12} className="mr-1 text-emerald-400" /> Mode
          </span>
          <span className="text-emerald-400 font-semibold flex items-center">
            <CheckCircle2 size={10} className="mr-1" /> 100% Offline
          </span>
        </div>
      </div>
    </div>
  );
};
