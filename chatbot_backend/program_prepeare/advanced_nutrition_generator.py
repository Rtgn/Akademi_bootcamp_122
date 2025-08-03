import os
import json
from datetime import datetime
from typing import Dict, Any
import google.generativeai as genai
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
import colorama
from colorama import Fore, Style

colorama.init()

class AdvancedNutritionGenerator:
    def __init__(self):
        self.console = Console()
        self.user_data = {}
        
        try:
            from config import Config
            
            if not Config.validate_config():
                self.console.print("[red]Please set your Gemini API key in config.py or as GEMINI_API_KEY environment variable[/red]")
                raise ValueError("Invalid configuration")
            
            self.gemini_api_key = Config.get_gemini_api_key()
        except ImportError as e:
            self.console.print(f"[red]Error importing config: {str(e)}[/red]")
            raise
        except Exception as e:
            self.console.print(f"[red]Error initializing nutrition generator: {str(e)}[/red]")
            raise
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        pass

    def _create_simple_llm(self):
        class SimpleLLM:
            def __init__(self, model):
                self.model = model
            
            def invoke(self, prompt):
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    return f"Error: {str(e)}"
            
            def bind(self, **kwargs):
                return self
        
        return SimpleLLM(self.model)

    def _analyze_health_data(self, user_data: Dict[str, Any]) -> str:
        analysis_prompt = f"""
        Analyze the following user health data and provide insights for nutrition planning:
        
        User Data: {json.dumps(user_data, indent=2)}
        
        Provide analysis on:
        1. Health risk factors
        2. Nutritional needs based on age, gender, and activity level
        3. Special considerations for pregnancy/postpartum if applicable
        4. Dietary restrictions and alternatives
        5. Recommended calorie and macronutrient ranges
        6. Important health notes for the nutrition program
        
        Format your response as a clear, structured analysis.
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            return response.text
        except Exception as e:
            return f"Health analysis error: {str(e)}"

    def _create_comprehensive_prompt(self, user_data: Dict[str, Any], health_analysis: str) -> str:
        prompt = f"""
        Create a comprehensive, personalized nutrition program based on the following information:
        
        USER HEALTH DATA:
        {json.dumps(user_data, indent=2)}
        
        HEALTH ANALYSIS:
        {health_analysis}
        
        REQUIREMENTS:
        1. Create a detailed, personalized nutrition program
        2. Include meal plans, recipes, and shopping lists
        3. Consider all health factors, dietary preferences, and restrictions
        4. Ensure the program is safe and medically appropriate
        5. Make it practical and easy to follow
        6. Include nutritional information and tips
        
        FORMAT:
        - Use clear headings and sections
        - Include meal plans with recipes
        - Provide shopping lists
        - Add nutritional tips and recommendations
        - Make it easy to read and follow
        
        IMPORTANT: Format your response as a well-structured, readable text document with clear sections, 
        not as JSON. Use headings, bullet points, and proper formatting to make it easy to read.
        """
        
        return prompt

    pass

    def _ask_health_questions(self) -> Dict[str, Any]:
        self.console.print(Panel.fit(
            "[bold blue]Welcome to the Advanced Nutrition Program Generator![/bold blue]\n"
            "I'll conduct a comprehensive health assessment to create your personalized nutrition program.",
            title="Health Assessment"
        ))

        basic_questions = [
            ("age", "What is your age?", int),
            ("weight", "What is your current weight (in kg)?", float),
            ("height", "What is your height (in cm)?", float),
            
            ("activity_level", "What is your activity level? (sedentary/light/moderate/active/very_active)", str),
            ("occupation", "What is your occupation? (desk_job/physical_work/mixed)", str),
            ("sleep_hours", "How many hours do you sleep per night?", int),
            ("stress_level", "What is your stress level? (low/medium/high)", str),
        ]
        
        pregnancy_questions = [
            ("pregnancy_weeks", "How many weeks pregnant are you?", int),
        ]
        
        postpartum_questions = [
            ("postpartum_weeks", "How many weeks ago did you give birth?", int),
            ("delivery_method", "What was your delivery method? (vaginal/caesarean/assisted)", str),
            ("breastfeeding", "Are you breastfeeding? (yes/no)", str),
            ("breastfeeding_weeks", "If breastfeeding, how many weeks have you been breastfeeding?", int),
        ]
        
        other_questions = [
            # Health Conditions
            ("allergies", "Do you have any food allergies? (list them or 'none')", str),
            ("medical_conditions", "Do you have any medical conditions? (diabetes/heart_disease/hypertension/none)", str),
            ("medications", "Are you taking any medications? (list them or 'none')", str),
            ("digestive_issues", "Do you have any digestive issues? (ibs/acid_reflux/none)", str),
            
            # Dietary Preferences
            ("dietary_preferences", "Any dietary preferences? (vegetarian/vegan/keto/paleo/mediterranean/none)", str),
            ("food_dislikes", "Any foods you dislike or avoid? (list them or 'none')", str),
            ("cooking_skills", "What are your cooking skills? (beginner/intermediate/advanced)", str),
            
            # Goals and Preferences
            ("goals", "What are your nutrition goals? (weight_loss/weight_gain/maintenance/health_improvement/energy_boost)", str),
            ("program_duration", "How many days would you like the program for? (default: 1)", int),
            ("meals_per_day", "How many meals per day do you prefer? (3/4/5/6)", int),
            ("snacks", "Do you want snacks included? (yes/no)", str),
            ("cooking_time", "How much time can you spend cooking per day? (quick/medium/elaborate)", str),
            ("budget", "What's your food budget level? (low/medium/high)", str),
            
            # Additional Preferences
            ("cuisine_preference", "Any cuisine preferences? (mediterranean/asian/italian/mexican/none)", str),
            ("spice_tolerance", "What's your spice tolerance? (mild/medium/hot)", str),
            ("meal_prep", "Do you prefer meal prep or daily cooking? (meal_prep/daily_cooking)", str)
        ]

        user_data = {}
        
        for field, question, data_type in basic_questions:
            while True:
                try:
                    answer = Prompt.ask(f"[bold green]{question}[/bold green]")
                    
                    if data_type == int:
                        user_data[field] = int(answer)
                    elif data_type == float:
                        user_data[field] = float(answer)
                    else:
                        user_data[field] = answer
                    break
                except ValueError:
                    self.console.print(f"[red]Please enter a valid {data_type.__name__}[/red]")
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Assessment cancelled by user.[/yellow]")
                    return None
        
        while True:
            try:
                pregnancy_status = Prompt.ask("[bold green]Are you currently pregnant? (yes/no)[/bold green]")
                if pregnancy_status.lower() in ['yes', 'no']:
                    user_data["pregnancy_status"] = pregnancy_status.lower()
                    break
                else:
                    self.console.print("[red]Please enter 'yes' or 'no'[/red]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Assessment cancelled by user.[/yellow]")
                return None
        
        if user_data["pregnancy_status"] == "yes":
            self.console.print("[bold blue]Pregnancy-specific questions:[/bold blue]")
            for field, question, data_type in pregnancy_questions:
                while True:
                    try:
                        answer = Prompt.ask(f"[bold green]{question}[/bold green]")
                        
                        if data_type == int:
                            user_data[field] = int(answer)
                        elif data_type == float:
                            user_data[field] = float(answer)
                        else:
                            user_data[field] = answer
                        break
                    except ValueError:
                        self.console.print(f"[red]Please enter a valid {data_type.__name__}[/red]")
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Assessment cancelled by user.[/yellow]")
                        return None
            
            user_data["postpartum"] = "no"
            user_data["postpartum_weeks"] = 0
            user_data["delivery_method"] = "not_applicable"
            user_data["breastfeeding"] = "no"
            user_data["breastfeeding_weeks"] = 0
            
        else:
            self.console.print("[bold blue]Postpartum questions:[/bold blue]")
            for field, question, data_type in postpartum_questions:
                while True:
                    try:
                        if field == "breastfeeding_weeks" and user_data.get("breastfeeding", "").lower() != "yes":
                            user_data[field] = 0
                            break
                        
                        answer = Prompt.ask(f"[bold green]{question}[/bold green]")
                        
                        if data_type == int:
                            user_data[field] = int(answer)
                        elif data_type == float:
                            user_data[field] = float(answer)
                        else:
                            user_data[field] = answer
                        break
                    except ValueError:
                        self.console.print(f"[red]Please enter a valid {data_type.__name__}[/red]")
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Assessment cancelled by user.[/yellow]")
                        return None
            
            user_data["pregnancy_weeks"] = 0
        
        for field, question, data_type in other_questions:
            while True:
                try:
                    if field == "program_duration" and not user_data.get(field):
                        user_data[field] = 1
                        break
                    
                    answer = Prompt.ask(f"[bold green]{question}[/bold green]")
                    
                    if data_type == int:
                        user_data[field] = int(answer)
                    elif data_type == float:
                        user_data[field] = float(answer)
                    else:
                        user_data[field] = answer
                    break
                except ValueError:
                    self.console.print(f"[red]Please enter a valid {data_type.__name__}[/red]")
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Assessment cancelled by user.[/yellow]")
                    return None

        return user_data

    pass

    def _call_gemini_api(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            raw_response = response.text
            
            formatted_response = self._format_response(raw_response)
            return formatted_response
        except Exception as e:
            return f"Error generating nutrition program: {str(e)}"

    def _format_response(self, raw_response: str) -> str:
        try:
            if raw_response.strip().startswith('{') and raw_response.strip().endswith('}'):
                import json
                data = json.loads(raw_response)
                return self._json_to_readable_format(data)
            else:
                return raw_response
        except (json.JSONDecodeError, KeyError):
            return raw_response

    def _json_to_readable_format(self, data: dict) -> str:
        formatted_text = []
        
        if 'introduction' in data:
            formatted_text.append("🍎 PERSONALIZED NUTRITION PROGRAM")
            formatted_text.append("=" * 50)
            formatted_text.append(data['introduction'])
            formatted_text.append("")
        
        if 'disclaimer' in data:
            formatted_text.append("⚠️  IMPORTANT DISCLAIMER")
            formatted_text.append("-" * 30)
            formatted_text.append(data['disclaimer'])
            formatted_text.append("")
        
        if 'nutrientRecommendations' in data:
            formatted_text.append("📋 NUTRIENT RECOMMENDATIONS")
            formatted_text.append("-" * 30)
            for nutrient, info in data['nutrientRecommendations'].items():
                nutrient_name = nutrient.replace('_', ' ').title()
                formatted_text.append(f"\n🔸 {nutrient_name}:")
                if 'sources' in info:
                    formatted_text.append("   Sources: " + ", ".join(info['sources']))
                if 'recommendations' in info:
                    formatted_text.append("   " + info['recommendations'])
            formatted_text.append("")
        
        # Meal Plan
        if 'mealPlan' in data:
            formatted_text.append("🍽️  DAILY MEAL PLAN")
            formatted_text.append("-" * 30)
            
            for day, meals in data['mealPlan'].items():
                day_name = day.title()
                formatted_text.append(f"\n📅 {day_name.upper()}")
                formatted_text.append("-" * 20)
                
                for meal_key, meal_info in meals.items():
                    if meal_key == 'snacks':
                        if meal_info:
                            formatted_text.append(f"\n🍎 Snacks:")
                            for snack in meal_info:
                                formatted_text.append(f"   • {snack}")
                    else:
                        meal_name = meal_info.get('name', 'Unknown Meal')
                        recipe = meal_info.get('recipe', 'No recipe provided')
                        calories = meal_info.get('calories', 'Calories not specified')
                        macros = meal_info.get('macros', {})
                        
                        formatted_text.append(f"\n🍽️  {meal_name}")
                        formatted_text.append(f"   Calories: {calories}")
                        if macros:
                            macro_text = []
                            if 'protein' in macros:
                                macro_text.append(f"Protein: {macros['protein']}g")
                            if 'carbs' in macros:
                                macro_text.append(f"Carbs: {macros['carbs']}g")
                            if 'fat' in macros:
                                macro_text.append(f"Fat: {macros['fat']}g")
                            if macro_text:
                                formatted_text.append(f"   Macros: {', '.join(macro_text)}")
                        formatted_text.append(f"   Recipe: {recipe}")
                
                formatted_text.append("")
        
        # Shopping List
        if 'shoppingList' in data:
            formatted_text.append("🛒 SHOPPING LIST")
            formatted_text.append("-" * 30)
            for item in data['shoppingList']:
                formatted_text.append(f"   • {item}")
            formatted_text.append("")
        
        # Meal Prep Guide
        if 'mealPrepGuide' in data:
            formatted_text.append("👩‍🍳 MEAL PREP GUIDE")
            formatted_text.append("-" * 30)
            formatted_text.append(data['mealPrepGuide'])
            formatted_text.append("")
        
        # Tips for Success
        if 'tipsForSuccess' in data:
            formatted_text.append("💡 TIPS FOR SUCCESS")
            formatted_text.append("-" * 30)
            formatted_text.append(data['tipsForSuccess'])
            formatted_text.append("")
        
        return "\n".join(formatted_text)

    def _save_program(self, username: str, program_content: str) -> str:
        """Save the nutrition program to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_program_{timestamp}.txt"
        
        # Ana dizine kaydet (main_chatbot.py'nin bulunduğu yer)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(program_content)
        
        return output_path

    def generate_program(self):
        try:
            self.console.print("\n[bold cyan]Step 1: Health Assessment[/bold cyan]")
            user_data = self._ask_health_questions()
            
            if user_data is None:
                return
            
            self.user_data = user_data
            
            self.console.print("\n[bold cyan]Step 2: Analyzing Health Data[/bold cyan]")
            health_analysis = self._analyze_health_data(user_data)
            
            self.console.print("\n[bold cyan]Step 3: Creating Nutrition Program[/bold cyan]")
            prompt = self._create_comprehensive_prompt(user_data, health_analysis)
            
            self.console.print("[yellow]Generating your personalized nutrition program...[/yellow]")
            program_content = self._call_gemini_api(prompt)
            
            from config import Config
            username = Config.DEFAULT_USERNAME
            filename = self._save_program(username, program_content)
            
            self.console.print(f"\n[bold green]✅ Nutrition Program Generated Successfully![/bold green]")
            self.console.print(f"[blue]Saved as: {filename}[/blue]\n")
            
            self.console.print(Panel(
                program_content,
                title="[bold]Your Personalized Nutrition Program[/bold]",
                border_style="green"
            ))
            
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")

def main():
    console = Console()
    
    console.print(Panel.fit(
        
        title="Welcome"
    ))
    
    generator = AdvancedNutritionGenerator()
    
    while True:
        try:
         
            console.print("1. Generate Nutrition Program")
            
        
           
            generator.generate_program()
            
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main() 