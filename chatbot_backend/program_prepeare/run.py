import sys
import os
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

def main():
    console = Console()
    
    try:
        console.print("\n[yellow]Starting Advanced Version...[/yellow]")
        try:
            import advanced_nutrition_generator
            # Ana chatbot'tan çağrıldığında otomatik olarak çalıştır
            generator = advanced_nutrition_generator.AdvancedNutritionGenerator()
            generator.generate_program()
        except ImportError as e:
            console.print(f"[red]Error: advanced_nutrition_generator.py not found![/red]")
            console.print(f"[red]Details: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main() 