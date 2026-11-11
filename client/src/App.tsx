import React, { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { MonacoEditor } from "./components/MonacoEditor";
import { AgentChatPane } from "./components/AgentChatPane";
import { CanvasPane } from "./components/CanvasPane";
import { TerminalPane } from "./components/TerminalPane";

export const App: React.FC = () => {
  const [currentFile, setCurrentFile] = useState<string>("pyproject.toml");
  const [activeView, setActiveView] = useState<string>("editor");

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#121316] select-none">
      {/* Left Sovereign Navigation & File Explorer */}
      <Sidebar
        currentFile={currentFile}
        onSelectFile={(path) => {
          setCurrentFile(path);
          setActiveView("editor");
        }}
        activeView={activeView}
        onSelectView={setActiveView}
      />

      {/* Center Dynamic Surface */}
      <div className="flex-1 flex overflow-hidden">
        {activeView === "editor" && <MonacoEditor currentFile={currentFile} />}
        {activeView === "canvas" && <CanvasPane />}
        {activeView === "terminal" && <TerminalPane />}
      </div>

      {/* Right Sovereign Agent ReAct Loop */}
      <AgentChatPane />
    </div>
  );
};
