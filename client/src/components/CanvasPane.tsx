import React, { useState, useEffect } from "react";
import {
  LayoutDashboard,
  FileText,
  GitCompare,
  Code,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  ExternalLink,
} from "lucide-react";
import { api } from "../services/api";
import { ArtifactMeta } from "../types";

export const CanvasPane: React.FC = () => {
  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [zoomLevel, setZoomLevel] = useState<number>(100);

  const fetchArtifacts = async () => {
    setLoading(true);
    try {
      const list = await api.listArtifacts();
      setArtifacts(list);
      if (list.length > 0 && !selectedId) {
        setSelectedId(list[0].id);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchArtifacts();
  }, []);

  const selectedArtifact = artifacts.find((a) => a.id === selectedId);

  const getBadgeIcon = (type: string) => {
    switch (type) {
      case "mermaid":
        return <Code size={12} className="mr-1 text-purple-400" />;
      case "diff":
        return <GitCompare size={12} className="mr-1 text-emerald-400" />;
      default:
        return <FileText size={12} className="mr-1 text-blue-400" />;
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#18191e] text-xs">
      {/* Canvas Header */}
      <div className="h-9 bg-[#141518] border-b border-[#2b2e37] flex items-center justify-between px-3">
        <div className="flex items-center space-x-2 font-semibold text-white">
          <LayoutDashboard size={14} className="text-purple-400" />
          <span>Interactive Artifacts Canvas</span>
        </div>
        <div className="flex items-center space-x-2">
          {selectedId && (
            <div className="flex items-center space-x-1 bg-[#1e2026] px-1.5 py-0.5 rounded border border-[#2d313b] text-[#9da3ae]">
              <button
                onClick={() => setZoomLevel((z) => Math.max(50, z - 10))}
                className="hover:text-white p-0.5"
                title="Zoom Out"
              >
                <ZoomOut size={12} />
              </button>
              <span className="text-[10px] font-mono px-1">{zoomLevel}%</span>
              <button
                onClick={() => setZoomLevel((z) => Math.min(200, z + 10))}
                className="hover:text-white p-0.5"
                title="Zoom In"
              >
                <ZoomIn size={12} />
              </button>
              <button
                onClick={() => setZoomLevel(100)}
                className="hover:text-white p-0.5 ml-1"
                title="Reset Zoom"
              >
                <Maximize2 size={11} />
              </button>
            </div>
          )}
          <button
            onClick={fetchArtifacts}
            className="text-[#9da3ae] hover:text-white p-1 rounded hover:bg-[#252830]"
            title="Refresh Artifacts"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left List of Artifacts */}
        <div className="w-64 border-r border-[#2b2e37] bg-[#141518] p-2 overflow-y-auto space-y-1">
          {artifacts.map((a) => (
            <div
              key={a.id}
              onClick={() => setSelectedId(a.id)}
              className={`p-2 rounded cursor-pointer transition-colors border ${
                selectedId === a.id
                  ? "bg-[#242834] border-blue-500/50 text-white"
                  : "bg-[#1b1d24] border-transparent text-[#9da3ae] hover:text-white hover:bg-[#20222a]"
              }`}
            >
              <div className="font-medium truncate text-[11px] mb-1">{a.title}</div>
              <div className="flex items-center justify-between text-[10px]">
                <span className="flex items-center">
                  {getBadgeIcon(a.artifact_type)}
                  <span className="uppercase">{a.artifact_type}</span>
                </span>
                <span className="text-[#64748b]">v{a.version}</span>
              </div>
            </div>
          ))}
          {artifacts.length === 0 && (
            <div className="text-center text-[#64748b] italic p-4">No artifacts generated</div>
          )}
        </div>

        {/* Right Preview Frame with Pan & Zoom */}
        <div className="flex-1 bg-[#121316] relative flex flex-col overflow-hidden">
          {selectedId ? (
            <div
              className="w-full h-full overflow-auto flex items-center justify-center p-2"
              style={{ background: "#0f1115" }}
            >
              <div
                style={{
                  transform: `scale(${zoomLevel / 100})`,
                  transformOrigin: "center center",
                  transition: "transform 0.15s ease-out",
                  width: "100%",
                  height: "100%",
                }}
              >
                <iframe
                  src={api.getArtifactHtmlUrl(selectedId)}
                  title="Artifact Preview"
                  className="w-full h-full border-none rounded shadow-2xl bg-[#1a1c23]"
                />
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-[#64748b]">
              Select an artifact from the list to preview
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
