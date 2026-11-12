import React, { useState, useEffect, useRef } from "react";
import Editor, { OnMount } from "@monaco-editor/react";
import { Save, Sparkles, Code2, AlertCircle } from "lucide-react";
import { api } from "../services/api";
import { Diagnostic } from "../types";

interface MonacoEditorProps {
  currentFile: string;
  onRunAgentOnCode?: (snippet: string) => void;
}

export const MonacoEditor: React.FC<MonacoEditorProps> = ({
  currentFile,
  onRunAgentOnCode,
}) => {
  const [content, setContent] = useState<string>("// Open a file from explorer to edit");
  const [loading, setLoading] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<string>("Ready");
  const [diagnosticsCount, setDiagnosticsCount] = useState<number>(0);
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);

  // Update diagnostics markers in Monaco
  const updateDiagnostics = async (filePath: string) => {
    if (!filePath || !editorRef.current || !monacoRef.current) return;
    try {
      const diags: Diagnostic[] = await api.getDiagnostics(filePath);
      const monaco = monacoRef.current;
      const model = editorRef.current.getModel();
      if (!model) return;

      const markers = diags.map((d) => ({
        severity:
          d.severity === 1
            ? monaco.MarkerSeverity.Error
            : monaco.MarkerSeverity.Warning,
        message: d.message,
        startLineNumber: Math.max(1, d.start_line),
        startColumn: Math.max(1, d.start_char + 1),
        endLineNumber: Math.max(1, d.end_line),
        endColumn: Math.max(1, d.end_char + 1),
      }));

      monaco.editor.setModelMarkers(model, "antimatter-lsp", markers);
      setDiagnosticsCount(markers.length);
    } catch {
      // Ignore diagnostics fetch errors gracefully
    }
  };

  // Load file content when currentFile changes
  useEffect(() => {
    if (!currentFile) return;
    setLoading(true);
    setStatusMsg(`Loading ${currentFile}...`);
    api
      .readFile(currentFile)
      .then((c) => {
        setContent(c);
        setStatusMsg("Loaded");
        setTimeout(() => updateDiagnostics(currentFile), 300);
      })
      .catch((err) => {
        setContent(`// Error loading file: ${err.message}`);
        setStatusMsg("Error");
      })
      .finally(() => setLoading(false));
  }, [currentFile]);

  // Handle Save
  const handleSave = async () => {
    if (!currentFile || !editorRef.current) return;
    setSaving(true);
    setStatusMsg("Saving...");
    try {
      const currentVal = editorRef.current.getValue();
      await api.writeFile(currentFile, currentVal);
      setStatusMsg("Saved");
      await updateDiagnostics(currentFile);
      setTimeout(() => setStatusMsg("Ready"), 2000);
    } catch (e: any) {
      setStatusMsg(`Save failed: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  // Cursor-style Inline Tab Completion (FIM)
  const triggerAutocomplete = async () => {
    if (!editorRef.current) return;
    const editor = editorRef.current;
    const position = editor.getPosition();
    const model = editor.getModel();

    const prefix = model.getValueInRange({
      startLineNumber: Math.max(1, position.lineNumber - 50),
      startColumn: 1,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    });

    const suffix = model.getValueInRange({
      startLineNumber: position.lineNumber,
      startColumn: position.column,
      endLineNumber: Math.min(model.getLineCount(), position.lineNumber + 20),
      endColumn: 1,
    });

    setStatusMsg("Generating FIM autocomplete...");
    const completion = await api.getAutocomplete(prefix, suffix, currentFile);
    if (completion) {
      editor.executeEdits("fim-autocomplete", [
        {
          range: {
            startLineNumber: position.lineNumber,
            startColumn: position.column,
            endLineNumber: position.lineNumber,
            endColumn: position.column,
          },
          text: completion,
        },
      ]);
      setStatusMsg("Completed");
    } else {
      setStatusMsg("No suggestion");
    }
  };

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Register Save command: Ctrl+S
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      handleSave();
    });

    // Register FIM Autocomplete command: Ctrl+Space
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Space, () => {
      triggerAutocomplete();
    });

    if (currentFile) {
      updateDiagnostics(currentFile);
    }
  };

  const detectLanguage = (path: string) => {
    if (path.endsWith(".py")) return "python";
    if (path.endsWith(".js") || path.endsWith(".jsx")) return "javascript";
    if (path.endsWith(".ts") || path.endsWith(".tsx")) return "typescript";
    if (path.endsWith(".json")) return "json";
    if (path.endsWith(".html")) return "html";
    if (path.endsWith(".md")) return "markdown";
    return "plaintext";
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#1e2026]">
      {/* Editor Tab Bar */}
      <div className="h-9 bg-[#141518] border-b border-[#2b2e37] flex items-center justify-between px-3">
        <div className="flex items-center space-x-2 text-xs">
          <Code2 size={14} className="text-blue-400" />
          <span className="font-mono text-white font-medium">{currentFile || "Untitled"}</span>
          {diagnosticsCount > 0 && (
            <span className="flex items-center text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded border border-amber-400/20">
              <AlertCircle size={10} className="mr-1" />
              {diagnosticsCount} issues
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={triggerAutocomplete}
            className="flex items-center px-2 py-1 text-xs bg-[#242832] hover:bg-[#2d3340] text-purple-300 rounded border border-[#3b4150]"
            title="Trigger Cursor FIM Completion (Ctrl+Space)"
          >
            <Sparkles size={12} className="mr-1 text-purple-400" /> Complete (FIM)
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center px-2.5 py-1 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded font-medium disabled:opacity-50"
            title="Save file (Ctrl+S)"
          >
            <Save size={12} className="mr-1" /> Save
          </button>
        </div>
      </div>

      {/* Monaco Core View */}
      <div className="flex-1 relative">
        {loading ? (
          <div className="flex items-center justify-center h-full text-sm text-[#9da3ae]">
            Loading code buffer...
          </div>
        ) : (
          <Editor
            height="100%"
            theme="vs-dark"
            language={detectLanguage(currentFile)}
            value={content}
            onMount={handleEditorDidMount}
            options={{
              fontSize: 13,
              fontFamily: "'JetBrains Mono', Consolas, monospace",
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 4,
              cursorBlinking: "smooth",
              renderLineHighlight: "all",
              lineNumbers: "on",
              glyphMargin: true,
            }}
          />
        )}
      </div>

      {/* Status Bar */}
      <div className="h-6 bg-[#141518] border-t border-[#2b2e37] flex items-center justify-between px-3 text-[11px] text-[#9da3ae]">
        <div className="flex items-center space-x-4 font-mono">
          <span>{detectLanguage(currentFile).toUpperCase()}</span>
          <span>UTF-8</span>
          <span>Diagnostics: {diagnosticsCount}</span>
        </div>
        <div className="text-right text-[#c2c7d0]">{statusMsg}</div>
      </div>
    </div>
  );
};
