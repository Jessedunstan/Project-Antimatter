"""Command Line Interface for Project Antimatter.

Provides rich terminal controls for Code Intelligence,
Vector Search, LSP diagnostics, and Autonomous Agent execution.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from antimatter.core.config import get_settings
from antimatter.context.service import CodeIntelligenceService
from antimatter.agent.loop import AgentReActLoop

console = Console()


@click.group()
def main():
    """Project Antimatter: Sovereign, 100% Offline Agentic Software Engineering Platform."""
    pass


@main.command()
def info():
    """Display runtime configuration, hardware profile, and sovereign offline status."""
    settings = get_settings()
    table = Table(title="[bold cyan]Antimatter Sovereign Engine Status[/bold cyan]")
    table.add_column("Property", style="bold magenta")
    table.add_column("Value", style="green")

    table.add_row("Version", settings.version)
    table.add_row("Profile", settings.profile.value.upper())
    table.add_row("LLM Backend", settings.llm_backend.value)
    table.add_row("Planner Model", settings.planner_model)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Workspace Directory", str(settings.workspace_dir))
    table.add_row("Telemetry", "[bold red]DISABLED (100% Offline)[/bold red]")

    console.print(table)


@main.command()
@click.argument("directory", required=False, type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, help="Re-index all files ignoring checksum cache")
def index(directory: str | None, force: bool):
    """Scan workspace, chunk code via Tree-sitter AST, and store embeddings in sqlite-vec."""
    settings = get_settings()
    target_dir = Path(directory or settings.workspace_dir).resolve()

    console.print(f"[bold cyan]Indexing workspace:[/bold cyan] {target_dir}")
    service = CodeIntelligenceService(workspace_dir=target_dir)

    with console.status("[bold green]Parsing ASTs and computing local embeddings..."):
        stats = service.index_workspace(force=force)

    console.print(
        Panel(
            f"[bold green]Indexing Complete![/bold green]\n"
            f"- Files Indexed: {stats['indexed_files']}\n"
            f"- Files Skipped (Unchanged): {stats['skipped_files']}\n"
            f"- Code Chunks Created: {stats['total_chunks_indexed']}\n"
            f"- Symbols Cataloged: {stats['total_symbols_indexed']}\n"
            f"- Duration: {stats['duration_sec']:.2f}s",
            title="Code Intelligence Index",
            border_style="green",
        )
    )


@main.command()
@click.argument("query")
@click.option("--limit", "-k", default=3, help="Number of nearest results to retrieve")
def search(query: str, limit: int):
    """Perform sub-second semantic search over code chunks using sqlite-vec."""
    service = CodeIntelligenceService()
    console.print(f"[bold cyan]Semantic Query:[/bold cyan] '{query}'")

    results = service.semantic_search(query=query, limit=limit)
    if not results:
        console.print("[yellow]No matching code chunks found.[/yellow]")
        return

    for i, res in enumerate(results, 1):
        chunk = res.chunk
        title = f"Match #{i} (Score: {res.score:.3f}) | {chunk.file_path}:{chunk.start_line}-{chunk.end_line} ({chunk.name})"
        syntax = Syntax(chunk.content, "python", theme="monokai", line_numbers=True, start_line=chunk.start_line)
        console.print(Panel(syntax, title=title, border_style="cyan"))


@main.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
def symbols(file_path: str):
    """Display Tree-sitter symbol hierarchy outline for a code file."""
    p = Path(file_path).resolve()
    service = CodeIntelligenceService()
    hierarchy = service.get_symbol_hierarchy(p)

    if not hierarchy:
        console.print(f"[yellow]No symbols found in '{file_path}'.[/yellow]")
        return

    root_tree = Tree(f"[bold green]{p.name}[/bold green] (Symbol Hierarchy)")

    def _add_nodes(tree_node, sym):
        lbl = f"[{sym.kind.value}] [bold cyan]{sym.name}[/bold cyan] (L{sym.start_line}-L{sym.end_line})"
        if sym.docstring:
            lbl += f" - [dim]{sym.docstring[:40]}...[/dim]"
        child_node = tree_node.add(lbl)
        for c in sym.children:
            _add_nodes(child_node, c)

    for sym in hierarchy:
        _add_nodes(root_tree, sym)

    console.print(root_tree)


@main.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
def check(file_path: str):
    """Run local LSP syntax and diagnostics check on a file."""
    p = Path(file_path).resolve()
    service = CodeIntelligenceService()

    async def _run_check():
        try:
            return await service.get_diagnostics(p)
        finally:
            await service.close()

    diags = asyncio.run(_run_check())
    if not diags:
        console.print(f"[bold green]No syntax issues or diagnostics in '{file_path}'.[/bold green]")
        return

    table = Table(title=f"Diagnostics for {p.name}")
    table.add_column("Severity", style="bold red")
    table.add_column("Line", style="cyan")
    table.add_column("Message", style="white")

    for d in diags:
        table.add_row(d.severity.name, str(d.line), d.message)

    console.print(table)


@main.command()
@click.argument("task")
@click.option("--max-steps", default=10, help="Maximum ReAct execution steps")
def run(task: str, max_steps: int):
    """Run sovereign ReAct agent to solve a software engineering task autonomously."""
    console.print(Panel(f"[bold white]{task}[/bold white]", title="Antimatter Agent Task", border_style="magenta"))

    def _step_callback(step):
        console.print(f"\n[bold yellow]Step {step.step_number}:[/bold yellow]")
        if step.thought:
            console.print(f"[dim]{step.thought}[/dim]")
        for tc in step.tool_calls:
            console.print(f"  [bold cyan]-> Invoking Tool:[/bold cyan] `{tc.name}`({tc.arguments})")
        for tid, res in step.tool_results.items():
            res_str = str(res)
            preview = res_str[:200] + ("..." if len(res_str) > 200 else "")
            console.print(f"  [green]<- Result:[/green] {preview}")

    agent = AgentReActLoop(max_steps=max_steps)

    with console.status("[bold green]Agent thinking and executing..."):
        trajectory = asyncio.run(agent.run(task=task, on_step=_step_callback))

    if trajectory.status.value == "completed":
        console.print("\n" + Panel(f"[bold green]{trajectory.final_response}[/bold green]", title="Task Completed", border_style="green"))
    else:
        console.print(f"\n[bold red]Agent Execution Failed:[/bold red] {trajectory.error_message}")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host bind address")
@click.option("--port", default=8000, help="Port to listen on")
def serve(host: str, port: int):
    """Start local Antimatter FastAPI REST & Streaming server."""
    import uvicorn
    console.print(f"[bold green]Starting Antimatter Server on {host}:{port}...[/bold green]")
    uvicorn.run("antimatter.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
