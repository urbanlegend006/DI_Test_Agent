import sys
import io
import logging
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except AttributeError:
        pass
    except (AttributeError, LookupError, UnicodeError) as e:
        logging.getLogger("reconciliation_agent.main").warning(
            "Failed to reconfigure stdout/stderr encoding: %s", e
        )

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.columns import Columns
from rich.text import Text

from config import OPENAI_MODEL, configure_logging

logger = logging.getLogger("reconciliation_agent.main")
console = Console()


def print_welcome_banner():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ra_art = Text(
        "██████╗     █████╗\n"
        "██╔══██╗   ██╔══██╗\n"
        "██████╔╝   ███████║\n"
        "██╔══██╗   ██╔══██║\n"
        "██║  ██║   ██║  ██║\n"
        "╚═╝  ╚═╝   ╚═╝  ╚═╝",
        style="bright_blue"
    )

    right_text = Text.assemble(
        ("ReconAgent  v1.0\n", "bold #6366f1"),
        ("Data Reconciliation CLI\n\n", "italic bright_blue"),
        ("\u2500" * 25 + "\n\n", "dim"),
        ("Model:  ", "bold white"), (OPENAI_MODEL + "\n", "bold bright_green"),
        ("Date:    ", "bold white"), (now + "\n\n", "bold bright_yellow"),
        ("\u2500" * 25 + "\n\n", "dim"),
        ("\U0001f916 I'm a Data Integrity Test Agent.\n", "bold cyan"),
        ("I handle all data formats \u2014 XLSX, CSV, JSON, TXT, Parquet \u2014\n", "white"),
        ("reconciling them row-by-row with precision.\n\n", "dim white"),
        ("Ready to start? ", "bright_white"),
        ("Share your Source & Target file paths and I'll do the rest.\n\n", "white"),
        ("Type /help for available commands", "dim italic"),
    )

    columns = Columns([ra_art, right_text], align="center", equal=False, expand=True)

    console.print(Panel(
        columns,
        border_style="#6366f1",
        padding=(1, 2),
        title="[bold #6366f1]ReconAgent[/bold #6366f1]",
    ))


def show_help():
    table = Table(
        title="[bold #6366f1]Available Commands[/bold #6366f1]",
        show_header=True,
        header_style="bold #6366f1",
        border_style="#6366f1",
    )
    table.add_column("Command", style="bold cyan", width=12)
    table.add_column("Description", style="white")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear screen and reset session state")
    table.add_row("/status", "Show current session status")
    table.add_row("/exit", "Exit the application")
    table.add_row("/quit", "Exit the application")
    console.print(table)


def show_status():
    from tools import SESSION_STATE

    src = SESSION_STATE.source_filename or "Not loaded"
    tgt = SESSION_STATE.target_filename or "Not loaded"
    has_recon = "Yes" if SESSION_STATE.reconciliation_results is not None else "No"

    table = Table(
        title="[bold #6366f1]Session Status[/bold #6366f1]",
        show_header=True,
        header_style="bold #6366f1",
        border_style="#6366f1",
    )
    table.add_column("Property", style="bold white", width=20)
    table.add_column("Value", style="bright_green")
    table.add_row("Model", OPENAI_MODEL)
    table.add_row("Source File", src)
    table.add_row("Target File", tgt)
    table.add_row("Reconciliation Run", has_recon)
    console.print(table)


def handle_slash_command(cmd: str) -> bool:
    from tools import SESSION_STATE

    cmd_lower = cmd.strip().lower()

    if cmd_lower in ("/exit", "/quit"):
        console.print("\n[bold red]Exiting chatbot loop. Goodbye![/bold red]")
        sys.exit(0)
    elif cmd_lower == "/help":
        show_help()
        return True
    elif cmd_lower == "/clear":
        SESSION_STATE.reset()
        console.clear()
        print_welcome_banner()
        return True
    elif cmd_lower == "/status":
        show_status()
        return True
    return False


def get_status_bar():
    from tools import SESSION_STATE

    src = SESSION_STATE.source_filename or "\u2014"
    tgt = SESSION_STATE.target_filename or "\u2014"
    return f"[dim]Model: {OPENAI_MODEL}  \u2502  Source: {src}  \u2502  Target: {tgt}[/dim]"


def print_reconciliation_table():
    from tools import SESSION_STATE

    if SESSION_STATE.reconciliation_results is None:
        return
    if SESSION_STATE._recon_table_shown:
        return
    SESSION_STATE._recon_table_shown = True

    results = SESSION_STATE.reconciliation_results
    summary = results['summary']

    table = Table(
        title="[bold #6366f1]Reconciliation Results Summary[/bold #6366f1]",
        show_header=True,
        header_style="bold #6366f1",
        border_style="#6366f1",
    )
    table.add_column("Reconciliation Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold green")

    table.add_row("Total Rows (Source)", str(summary['total_source_rows']))
    table.add_row("Total Rows (Target)", str(summary['total_target_rows']))
    table.add_row("Fully Matched Rows", str(summary['matched_rows']))
    table.add_row("Mismatched Rows", str(summary['mismatched_rows']))
    table.add_row("Missing in Target", str(summary['missing_in_target']))
    table.add_row("Missing in Source", str(summary['missing_in_source']))

    if summary['duplicate_source'] > 0:
        table.add_row("Duplicates (Source)", str(summary['duplicate_source']), style="yellow")
    if summary['duplicate_target'] > 0:
        table.add_row("Duplicates (Target)", str(summary['duplicate_target']), style="yellow")

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    """Entry point for the ReconAgent CLI chatbot.

    Configures logging and library loggers, initialises the agent, then runs
    the interactive prompt loop with slash-command and error handling.
    """
    configure_logging()

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    from agent import get_reconciliation_agent

    print_welcome_banner()

    try:
        agent_executor = get_reconciliation_agent()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize agent executor: {str(e)}[/bold red]")
        logger.exception("Agent initialization error")
        sys.exit(1)

    while True:
        try:
            user_input = console.input("[bold cyan]  \u276f[/bold cyan] ")

            if not user_input.strip():
                continue

            if user_input.strip().startswith("/"):
                if handle_slash_command(user_input.strip()):
                    continue

            user_input_clean = user_input.strip().lower()
            if user_input_clean in ("exit", "quit", "q"):
                console.print("\n[bold red]Exiting chatbot loop. Goodbye![/bold red]")
                sys.exit(0)

            console.print("[bold green]Agent thinking...[/bold green]")
            response = agent_executor.invoke({"input": user_input})
            agent_response = response.get("output", "No response received.")

            console.print(Markdown(f" \U0001f916 {agent_response}"))

            print_reconciliation_table()

            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold red]Process interrupted by user. Exiting...[/bold red]")
            sys.exit(0)

        except Exception as e:
            error_msg = str(e)
            logger.exception("Error in CLI chat loop")

            if "AuthenticationError" in type(e).__name__ or "401" in error_msg or "api_key" in error_msg.lower():
                console.print(Panel(
                    "[bold red]Authentication Failed[/bold red]\n\n"
                    "Your OpenAI API key is invalid or has expired.\n"
                    "Please update the [bold cyan]OPENAI_API_KEY[/bold cyan] in your [bold cyan].env[/bold cyan] file "
                    "with a valid key from [link=https://platform.openai.com/account/api-keys]platform.openai.com[/link].",
                    title="[bold red]\u26a0 API Key Error[/bold red]",
                    border_style="red"
                ))
            else:
                console.print(f"\n[bold red]An unexpected error occurred: {error_msg}[/bold red]")
                console.print("[yellow]Please check the logs at logs/app.log for details.[/yellow]")

            console.print()


if __name__ == "__main__":
    main()
