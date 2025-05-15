"""
Generates an HTML report from metrics data stored by MetricsStore.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Assuming MetricsStore and MetricEvent are accessible.
# This might require adjusting sys.path or installing chungoid-core as a package.
try:
    from chungoid.utils.metrics_store import MetricsStore, MetricEvent
    from chungoid.schemas.metrics import MetricEventType # For potential type checking/filtering
except ImportError:
    # Fallback for direct script execution if chungoid is not in PYTHONPATH
    # This is a common pattern for utility scripts.
    # You might need to adjust the number of .parent calls based on your script's location
    # relative to the src directory.
    import sys
    core_src_path = Path(__file__).resolve().parent.parent / 'src'
    if core_src_path.exists() and str(core_src_path) not in sys.path:
        sys.path.insert(0, str(core_src_path))
    from chungoid.utils.metrics_store import MetricsStore, MetricEvent
    from chungoid.schemas.metrics import MetricEventType


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chungoid Metrics Report{title_suffix}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1, h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 10px;}}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1);}}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #e9e9e9; }}
        .summary {{ margin-bottom: 30px; padding: 15px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1);}}
        .summary p {{ margin: 5px 0; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #777; }}
        .event-data {{ max-width: 300px; overflow-wrap: break-word; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Chungoid Metrics Report{title_suffix}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        {summary_section}
    </div>

    <h2>Metric Events</h2>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Type</th>
                <th>Run ID</th>
                <th>Flow ID</th>
                <th>Stage ID</th>
                <th>Master Stage ID</th>
                <th>Agent ID</th>
                <th>Data/Details</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    <div class="footer">
        Report generated at {generation_time}
    </div>
</body>
</html>
"""

def generate_html_table_rows(events: List[MetricEvent]) -> str:
    rows_html = []
    for event in events:
        data_str = json.dumps(event.data, indent=2, sort_keys=True) if event.data else "-"
        # Truncate very long data strings for display
        if len(data_str) > 200:
            data_str = data_str[:197] + "..."
        
        rows_html.append(f"""
            <tr>
                <td>{event.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if event.timestamp else '-'}</td>
                <td>{event.event_type.value if event.event_type else '-'}</td>
                <td>{event.run_id or '-'}</td>
                <td>{event.flow_id or '-'}</td>
                <td>{event.stage_id or '-'}</td>
                <td>{event.master_stage_id or '-'}</td>
                <td>{event.agent_id or '-'}</td>
                <td class="event-data"><pre>{data_str}</pre></td>
            </tr>
        """)
    return "\n".join(rows_html)

def generate_summary_section(events: List[MetricEvent], target_run_id: Optional[str]) -> str:
    if not events:
        return "<p>No metric events found to generate a summary.</p>"

    # For simplicity, if a specific run_id is targeted, summary is for that run.
    # Otherwise, it's a more general summary or a summary of the first run found.
    # This can be expanded later.
    
    summary_parts = []
    
    if target_run_id:
        run_events = [e for e in events if e.run_id == target_run_id]
        if not run_events:
             return f"<p>No events found for Run ID: {target_run_id}</p>"
        
        flow_id = run_events[0].flow_id if run_events[0].flow_id else "N/A"
        summary_parts.append(f"<p><b>Run ID:</b> {target_run_id}</p>")
        summary_parts.append(f"<p><b>Flow ID:</b> {flow_id}</p>")

        flow_start_event = next((e for e in run_events if e.event_type == MetricEventType.FLOW_START), None)
        flow_end_event = next((e for e in reversed(run_events) if e.event_type == MetricEventType.FLOW_END), None) # Get last one

        start_time = flow_start_event.timestamp if flow_start_event else None
        end_time = flow_end_event.timestamp if flow_end_event else None
        
        summary_parts.append(f"<p><b>Start Time:</b> {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'}</p>")
        summary_parts.append(f"<p><b>End Time:</b> {end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'}</p>")

        duration_str = "N/A"
        if start_time and end_time:
            duration = end_time - start_time
            duration_str = str(duration)
        elif flow_end_event and flow_end_event.data and "total_duration_seconds" in flow_end_event.data:
             duration_str = f"{flow_end_event.data['total_duration_seconds']:.2f}s (from event)"
        summary_parts.append(f"<p><b>Duration:</b> {duration_str}</p>")
        
        status = flow_end_event.data.get("final_status", "INCOMPLETE") if flow_end_event and flow_end_event.data else "INCOMPLETE"
        summary_parts.append(f"<p><b>Overall Status:</b> {status}</p>")

        stage_events = [e for e in run_events if e.event_type == MetricEventType.STAGE_END and e.data]
        num_stages = len(set(e.stage_id for e in run_events if e.stage_id))
        succeeded_stages = sum(1 for e in stage_events if e.data.get("status") == "COMPLETED_SUCCESS")
        failed_stages = sum(1 for e in stage_events if e.data.get("status") == "COMPLETED_FAILURE")
        
        summary_parts.append(f"<p><b>Total Unique Stages Encountered:</b> {num_stages}</p>")
        summary_parts.append(f"<p><b>Stages Succeeded:</b> {succeeded_stages}</p>")
        summary_parts.append(f"<p><b>Stages Failed:</b> {failed_stages}</p>")

    else: # General summary if no specific run_id
        num_total_events = len(events)
        run_ids = set(e.run_id for e in events if e.run_id)
        summary_parts.append(f"<p><b>Total Events Displayed:</b> {num_total_events}</p>")
        summary_parts.append(f"<p><b>Unique Run IDs in Displayed Events:</b> {len(run_ids)}</p>")
        if len(run_ids) == 1:
             # If only one run_id in the selection, show more details for it
             return generate_summary_section(events, list(run_ids)[0])


    return "\n".join(summary_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML metrics report for Chungoid.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory containing the .chungoid folder (default: current directory)."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific Run ID to generate the report for. If not provided, shows all events."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None, # Will default to project_dir/.chungoid/reports
        help="Directory to save the HTML report. (default: project_dir/.chungoid/reports/)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000, # Default to a higher limit for reports
        help="Maximum number of events to include in the report (default: 1000)."
    )

    args = parser.parse_args()

    project_root = args.project_dir.resolve()
    
    output_dir = args.output_dir
    if not output_dir:
        output_dir = project_root / ".chungoid" / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics_store = MetricsStore(project_root=project_root)
        # Fetch events, sorted ascending for chronological display in the report table
        events = metrics_store.get_events(run_id=args.run_id, limit=args.limit, sort_desc=False)
    except Exception as e:
        print(f"Error accessing MetricsStore or getting events: {e}")
        sys.exit(1)

    if not events:
        print(f"No metric events found for Run ID '{args.run_id if args.run_id else 'all'}'. Report not generated.")
        return

    table_rows_html = generate_html_table_rows(events)
    summary_html = generate_summary_section(events, args.run_id)
    
    title_suffix = f" - Run {args.run_id}" if args.run_id else " - All Runs"
    generation_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')

    html_content = HTML_TEMPLATE.format(
        title_suffix=title_suffix,
        summary_section=summary_html,
        table_rows=table_rows_html,
        generation_time=generation_time_str
    )
    
    report_filename_suffix = f"run_{args.run_id}" if args.run_id else "all_runs"
    report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file_path = output_dir / f"metrics_report_{report_filename_suffix}_{report_timestamp}.html"

    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully generated metrics report: {report_file_path}")
    except IOError as e:
        print(f"Error writing HTML report to {report_file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 