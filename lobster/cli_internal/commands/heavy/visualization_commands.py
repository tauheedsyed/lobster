"""
Shared visualization commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter
from lobster.utils import open_path

logger = logging.getLogger(__name__)


def _legacy_plot_names(data_manager) -> Optional[list]:
    """Display-safe names of the flat (legacy) files in plots/; ``None`` if unreadable."""
    from lobster.core.figures.listing import legacy_plot_files, safe_label

    try:
        names = legacy_plot_files(data_manager.workspace_path)
    except Exception:
        return None
    return sorted(safe_label(n) or "unnamed" for n in names)


def _print_legacy_not_supported(names, output, where: str) -> None:
    """Say plainly that legacy plot files exist and are not supported here (read-only)."""
    if names is None:
        output.print(
            "[yellow]⚠ Legacy plot files in plots/ could not be listed.[/yellow]",
            style="warning",
        )
    elif names:
        output.print(
            f"[yellow]⚠ {len(names)} legacy plot file(s) in plots/ are not supported by "
            f"{where} here; they are not included and stay unchanged.[/yellow]",
            style="warning",
        )


def _print_not_copied(data_manager, output) -> None:
    """Make anything the package could not copy visible here, not only in index.json."""
    report = getattr(data_manager, "last_package_report", None)
    if not isinstance(report, dict):
        return
    legacy = report.get("legacy_plot_files_not_copied")  # None = could not be listed
    missing = (
        int(report.get("figures_not_exported") or 0),
        int(report.get("html_views_not_written") or 0),
        int(report.get("recorded_png_not_copied") or 0),
    )
    legacy_text = (
        "legacy plot files could not be listed"
        if legacy is None
        else f"{int(legacy)} legacy plot file(s)"
    )
    if any(missing) or legacy is None or legacy:
        output.print(
            f"[yellow]⚠ Not copied: {missing[0]} figure revision(s), {missing[1]} HTML "
            f"view(s), {missing[2]} recorded PNG(s), {legacy_text}. "
            "See README.md 'Not copied'.[/yellow]",
            style="warning",
        )


def export_data(
    client: "AgentClient",
    output: OutputAdapter,
    include_png: bool = True,
    force_resave: bool = False,
    console=None,
) -> Optional[str]:
    """
    Export session data and plots to workspace exports directory.

    Exports all modalities and generated plots to workspace/exports/.
    For local clients with data, creates a comprehensive data package.
    For cloud clients or sessions without data, exports session metadata.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        include_png: ``False`` (``--no-png``) disables PNG generation only; no new PNG is
            rendered locally anyway. The package holds each saved revision's exact bytes,
            an HTML view and every already recorded PNG rendition; the index records
            ``png: not_rendered_locally`` (``None`` when False).
        force_resave: Force re-serialization of all modalities (default: False)
        console: Optional Rich console for progress display

    Returns:
        Summary string for conversation history, or None
    """
    try:
        # Check if this is a local client with detailed export capabilities
        if hasattr(client, "data_manager") and hasattr(
            client.data_manager, "create_data_package"
        ):
            # For local client, check what we're exporting
            data_manager = client.data_manager
            has_data = data_manager.has_data()
            saved_figures = len(data_manager._saved_figure_records())
            has_plots = bool(saved_figures)

            if has_data:
                modality_count = len(data_manager.modalities)
                plot_count = saved_figures

                # Build progress message
                items = []
                if modality_count:
                    items.append(f"{modality_count} dataset(s)")
                if plot_count:
                    items.append(f"{plot_count} plot(s)")

                # Use Rich progress if console is available
                if console is not None:
                    export_stats = {"reused": 0, "saved": 0, "plots": 0}

                    with console.status(
                        f"[bold cyan]Exporting {', '.join(items)}...[/bold cyan]",
                        spinner="dots",
                    ) as status:

                        def update_progress(msg: str):
                            # Parse message for statistics
                            if "Reused existing" in msg:
                                export_stats["reused"] += 1
                            elif "Saving modality" in msg:
                                export_stats["saved"] += 1
                            elif "Saving plot" in msg:
                                export_stats["plots"] += 1
                            status.update(f"[bold cyan]{msg}[/bold cyan]")

                        export_path = data_manager.create_data_package(
                            output_dir=str(data_manager.exports_dir),
                            progress_callback=update_progress,
                            include_png=include_png,
                            reuse_autosave=not force_resave,
                        )
                        export_path = Path(export_path)

                    # Show summary
                    output.print("[green]✓ Export complete:[/green]", style="success")
                    _print_not_copied(data_manager, output)
                    output.print(f"  [green]•[/green] {export_path.name}", style="info")
                    if export_stats["reused"] > 0:
                        output.print(
                            f"  [dim]⚡ Reused {export_stats['reused']} cached file(s)[/dim]",
                            style="info",
                        )
                else:
                    # Fallback: simple progress callback
                    output.print(
                        f"[yellow]Exporting {', '.join(items)}...[/yellow]",
                        style="info",
                    )

                    def simple_progress(msg: str):
                        logger.debug(f"Export progress: {msg}")

                    export_path = data_manager.create_data_package(
                        output_dir=str(data_manager.exports_dir),
                        progress_callback=simple_progress,
                        include_png=include_png,
                        reuse_autosave=not force_resave,
                    )
                    export_path = Path(export_path)
                    output.print(
                        f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                        style="success",
                    )
                    _print_not_copied(data_manager, output)
            else:
                # Fallback to regular export_session for non-data exports
                if has_plots:
                    plot_count = saved_figures
                    output.print(
                        f"[yellow]Exporting {plot_count} plot(s)...[/yellow]",
                        style="info",
                    )
                export_path = client.export_session()
                output.print(
                    f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                    style="success",
                )
                _print_legacy_not_supported(
                    _legacy_plot_names(data_manager), output, "export"
                )
        else:
            # For cloud client or other clients
            output.print(
                "[yellow]Exporting session data and plots...[/yellow]", style="info"
            )
            export_path = client.export_session()
            output.print(
                f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                style="success",
            )

        return f"Session exported to: {export_path}"

    except Exception as e:
        output.print(f"[red]Export failed: {e}[/red]", style="error")
        logger.exception("Export error")
        return f"Export failed: {e}"


def plots_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show list of generated plots with IDs, titles, sources, and timestamps.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        history, stats, _ = client.data_manager.plot_manager.get_plot_history()

        if history:
            output.print("\n[bold red]📊 Generated Plots[/bold red]\n", style="info")

            table_data = {
                "title": "🦞 Generated Plots",
                "columns": [
                    {"name": "ID", "style": "bold white"},
                    {"name": "Title", "style": "white"},
                    {"name": "Source", "style": "grey74"},
                    {"name": "Created", "style": "grey50"},
                ],
                "rows": [],
            }

            for plot in history:
                try:
                    created = datetime.fromisoformat(
                        plot["timestamp"].replace("Z", "+00:00")
                    )
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created_str = plot["timestamp"][:16] if plot["timestamp"] else "N/A"

                table_data["rows"].append(
                    [plot["id"], plot["title"], plot["source"] or "N/A", created_str]
                )

            output.print_table(table_data)

            # Add usage hint
            output.print(
                "\n[grey50]Tip: Use /plot <ID> to open a specific plot[/grey50]",
                style="info",
            )
            _print_legacy_not_supported(
                _legacy_plot_names(client.data_manager), output, "/plot or /export"
            )

            return f"Displayed {len(history)} plot(s)"
        else:
            # This session made none; the workspace may still hold saved or legacy plots.
            data_manager = client.data_manager
            legacy = _legacy_plot_names(data_manager)
            try:
                records = data_manager._saved_figure_records()
                # One record per revision; a figure can have several revisions.
                saved = len(records)
                figures = len({r["figure_id"] for r in records})
            except Exception as e:
                from lobster.core.figures.errors import safe_reason

                output.print(
                    f"[yellow]⚠ Saved figures could not be read ({safe_reason(e)}).[/yellow]",
                    style="warning",
                )
                _print_legacy_not_supported(legacy, output, "/plot or /export")
                return "Saved figures could not be read"
            if not saved and not legacy and legacy is not None:
                output.print("[grey50]No plots generated yet[/grey50]", style="info")
                return "No plots available"
            if saved:
                output.print(
                    f"[grey50]No plots generated in this session; {figures} saved "
                    f"figure(s), {saved} revision(s) in this workspace. Use /plot "
                    "<figure ref or title> to open one.[/grey50]",
                    style="info",
                )
            _print_legacy_not_supported(legacy, output, "/plot or /export")
            return (
                f"No session plots; {figures} saved figure(s), {saved} revision(s) "
                "in the workspace"
            )

    except Exception as e:
        output.print(f"[red]Failed to retrieve plot history: {e}[/red]", style="error")
        logger.exception("Plot list error")
        return None


def plot_show(
    client: "AgentClient", output: OutputAdapter, plot_identifier: Optional[str] = None
) -> Optional[str]:
    """
    Show or open a specific plot by ID or title match.

    If no plot_identifier is provided, opens the plots directory in file manager.
    If plot_identifier is provided, searches for plot by:
      1. Exact ID match
      2. Partial title match (case-insensitive)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        plot_identifier: Plot ID or partial title to search for. None opens plots directory.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        from lobster.core.figures.renditions import (
            describe_renditions,
            ensure_html_rendition,
            ensure_html_renditions,
        )

        data_manager = client.data_manager
        plots_dir = data_manager.workspace_path / "plots"
        store = data_manager.figure_store

        # Case 1: No identifier - write missing HTML views (explicit view path), open the dir
        if not plot_identifier:
            plots_dir.mkdir(parents=True, exist_ok=True)
            results = ensure_html_renditions(store)
            if results:
                failed = any(r.status != "available" for r in results)
                output.print(
                    f"[{'yellow' if failed else 'green'}]{describe_renditions(results)}"
                    f"[/{'yellow' if failed else 'green'}]",
                    style="warning" if failed else "success",
                )

            success, message = open_path(plots_dir)

            if success:
                output.print(f"[green]✓ {message}[/green]", style="success")
                return f"Opened plots directory: {plots_dir}"
            else:
                output.print(
                    f"[red]Failed to open plots directory: {message}[/red]",
                    style="error",
                )
                output.print(
                    f"[grey50]Plots directory: {plots_dir}[/grey50]", style="info"
                )
                return f"Failed to open plots directory: {message}"

        # Case 2: Identifier provided - session plot ID/title, or a saved figure ref/id
        ref, title = _resolve_figure(data_manager, plot_identifier)
        if ref is None:
            legacy = _legacy_match(data_manager, plot_identifier)
            if legacy is _LEGACY_UNLISTED:
                output.print(
                    f"[yellow]{plot_identifier} is not among saved figures; legacy plot "
                    "files could not be listed.[/yellow]",
                    style="warning",
                )
                return (
                    f"Plot not among saved figures: {plot_identifier}; legacy plot "
                    "files could not be listed"
                )
            if legacy:
                output.print(
                    f"[yellow]{legacy} is a legacy plot file in plots/ (read-only). "
                    "Opening legacy plot files is not supported here; it is "
                    "unchanged.[/yellow]",
                    style="warning",
                )
                return f"Legacy plot file not supported here: {legacy}"
            output.print(f"[red]Plot not found: {plot_identifier}[/red]", style="error")
            output.print(
                "[grey50]Use /plots to see available plot IDs and titles[/grey50]",
                style="info",
            )
            return f"Plot not found: {plot_identifier}"

        view = ensure_html_rendition(store, ref)
        if view.status == "pending":
            output.print(
                f"[yellow]Figure {ref}: save confirmation pending ({view.reason}).[/yellow]",
                style="warning",
            )
            return f"Figure {ref}: save confirmation pending: {view.reason}"
        if view.status != "available":
            output.print(
                f"[yellow]Figure {ref} is saved; its HTML view failed ({view.reason}).[/yellow]",
                style="warning",
            )
            return f"Figure {ref} saved; HTML view failed: {view.reason}"

        html_path = data_manager.workspace_path / view.path
        success, message = open_path(html_path)
        if success:
            output.print(
                f"[green]✓ Opened plot:[/green] [grey74]{title}[/grey74] [grey50]({ref})[/grey50]",
                style="success",
            )
            return f"Opened plot: {title} ({ref})"
        output.print(f"[red]Failed to open plot: {message}[/red]", style="error")
        output.print(f"[grey50]Plot file: {html_path}[/grey50]", style="info")
        return f"Failed to open plot: {message}"

    except Exception as e:
        output.print(f"[red]Failed to show plot: {e}[/red]", style="error")
        logger.exception("Plot show error")
        return None


_LEGACY_UNLISTED = object()  # the legacy files could not be listed (never "not found")


def _legacy_match(data_manager, identifier: str):
    """The display-safe name of the legacy plot file ``identifier`` names (file or stem);
    ``None`` if none does; ``_LEGACY_UNLISTED`` if the legacy files cannot be listed."""
    from lobster.core.figures.listing import legacy_plot_files, safe_label

    needle = identifier.strip()
    try:
        names = legacy_plot_files(data_manager.workspace_path)
    except Exception:
        return _LEGACY_UNLISTED
    for name in sorted(names):
        if needle in (name, Path(name).stem):
            return safe_label(name) or "unnamed"
    return None


def _resolve_figure(data_manager, identifier: str):
    """Map a session plot ID / title / figure ref / figure ID to ``(ref, title)`` (restart-safe)."""
    needle = identifier.strip()
    for entry in reversed(data_manager.latest_plots):
        if entry.get("id") == needle or (
            needle.lower() in str(entry.get("title", "")).lower()
            or needle.lower() in str(entry.get("original_title", "")).lower()
        ):
            if entry.get("figure_ref"):
                return entry["figure_ref"], entry.get("original_title") or entry.get(
                    "title"
                )
    store = data_manager.figure_store
    cursor, newest = None, None
    while True:
        page = store.list(cursor=cursor, limit=500)
        for rec in page.items:
            ref = f"{rec['figure_id']}@{rec['revision_id']}"
            label = rec.get("label") or ""
            if needle in (ref, rec["figure_id"]) or (
                label and needle.lower() in label.lower()
            ):
                newest = (
                    ref,
                    label or ref,
                )  # list is oldest→newest: keep the newest match
        cursor = page.next_cursor
        if not cursor:
            break
    return newest if newest else (None, None)
