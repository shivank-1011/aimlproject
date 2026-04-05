import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)

    def generate_study_plan(self, student_performance, goal):
        """
        Generates a student performance diagnosis and a personalized study plan.
        
        Args:
            student_performance (dict): A dictionary containing student performance metrics (e.g., subjects and scores).
            goal (str): The student's academic or career goal.
            
        Returns:
            str: The LLM's response containing diagnosis and study plan.
        """
        
        system_prompt = """
        You are an expert academic counselor and tutor. Your task is to analyze student performance data,
        provide a detailed diagnosis of their strengths and weaknesses, and create a structured, 
        actionable study plan to help them reach their specific goals.
        
        Format your response clearly with the following sections:
        1. **Performance Diagnosis**: A detailed breakdown of where the student stands.
        2. **Strengths & Weaknesses**: Bullet points highlighting key areas.
        3. **Personalized Study Plan**: A step-by-step plan (weekly or daily) to achieve their goal.
        4. **Recommended Resources**: Suggestions for books, courses, or practice tools.
        """
        
        user_prompt = f"""
        **Student Performance Data:**
        {student_performance}
        
        **Student Goal:**
        {goal}
        
        Please provide a comprehensive diagnosis and a tailored study plan.
        """
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error connecting to Groq API: {str(e)}"

if __name__ == "__main__":
    # Example usage
    try:
        analyzer = GroqAnalyzer()
        
        # Sample data
        performance = {
            "Maths": 45,
            "SESD": 78,
            "AIML": 62,
            "FSD": 85,
            "DVA": 55,
            "Overall Attendance": "70%"
        }
        goal = "Get a high-paying placement in a Top Tech company as an AI/ML Engineer."
        
        print("--- Analyzing Student Performance ---")
        diagnosis = analyzer.generate_study_plan(performance, goal)
        print(diagnosis)
        
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
