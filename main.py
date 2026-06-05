import sys
import io
import logging
import warnings

# Suppress deprecation warnings from LangChain and other libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 console output on Windows to handle emojis correctly
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass  # Fallback if stream is not bufferable or already reconfigured

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

# Initialize configurations & logging first
import config

# Silence noisy third-party loggers that clutter the chat UI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

from agent import get_reconciliation_agent

logger = logging.getLogger("reconciliation_agent.main")
console = Console()

def print_welcome_banner():
    """Prints a beautiful welcome banner to the terminal."""
    banner_text = (
        "[bold #6366f1]Data Reconciliation CLI Test Agent[/bold #6366f1]\n"
        "[dim]Version 1.0.0[/dim]\n\n"
        "Welcome! I am your reconciliation assistant. I can help you reconcile "
        "tabular data (Excel, JSON, or CSV) row-by-row and generate highly "
        "interactive visual HTML reports.\n\n"
        "To get started, simply provide the file paths of your [bold cyan]Source[/bold cyan] and [bold cyan]Target[/bold cyan] files.\n"
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to close the chat loop."
    )
    console.print(Panel(
        banner_text,
        title="[bold #6366f1]ReconAgent[/bold #6366f1]",
        border_style="#6366f1",
        padding=(1, 2)
    ))

def print_reconciliation_table():
    """Attempts to print a summary table of the latest reconciliation results if available."""
    from tools import SESSION_STATE
    
    if 'reconciliation_results' not in SESSION_STATE:
        return
        
    results = SESSION_STATE['reconciliation_results']
    summary = results['summary']
    
    table = Table(title="Reconciliation Results Summary", show_header=True, header_style="bold #6366f1")
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
    print_welcome_banner()
    
    # Initialize the LangChain Agent
    try:
        agent_executor = get_reconciliation_agent()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize agent executor: {str(e)}[/bold red]")
        logger.exception("Agent initialization error")
        sys.exit(1)
        
    while True:
        try:
            # Get User Input
            user_input = Prompt.ask("\n[bold cyan]User[/bold cyan]")
            
            # Clean and check for exit commands
            user_input_clean = user_input.strip().lower()
            if user_input_clean in ("exit", "quit", "q"):
                console.print("\n[bold red]Exiting chatbot loop. Goodbye![/bold red]")
                sys.exit(0)
                
            if not user_input.strip():
                continue
                
            # Process prompt with agent status spinner
            with Status("[bold green]Agent thinking...[/bold green]", spinner="dots", console=console):
                # Call agent executor
                response = agent_executor.invoke({"input": user_input})
                agent_response = response.get("output", "No response received.")
                
            # Print Agent text output
            console.print(f"\n[bold green]Agent:[/bold green] {agent_response}")
            
            # Proactively show results table if reconciliation was run in this turn
            print_reconciliation_table()
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Process interrupted by user. Exiting...[/bold red]")
            sys.exit(0)
            
        except Exception as e:
            error_msg = str(e)
            logger.exception("Error in CLI chat loop")
            
            # Check for common authentication errors
            if "AuthenticationError" in type(e).__name__ or "401" in error_msg or "api_key" in error_msg.lower():
                console.print(Panel(
                    "[bold red]Authentication Failed[/bold red]\n\n"
                    "Your OpenAI API key is invalid or has expired.\n"
                    "Please update the [bold cyan]OPENAI_API_KEY[/bold cyan] in your [bold cyan].env[/bold cyan] file "
                    "with a valid key from [link=https://platform.openai.com/account/api-keys]platform.openai.com[/link].",
                    title="[bold red]⚠ API Key Error[/bold red]",
                    border_style="red"
                ))
            else:
                console.print(f"\n[bold red]An unexpected error occurred: {error_msg}[/bold red]")
                console.print("[yellow]Please check the logs at logs/app.log for details.[/yellow]")

if __name__ == "__main__":
    main()
