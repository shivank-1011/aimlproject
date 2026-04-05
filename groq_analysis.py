import os
from groq import Groq
from dotenv import load_dotenv
from diagnosis import StudentDiagnosis

load_dotenv()

class GroqAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.diagnosis_tool = StudentDiagnosis()

    def generate_study_plan(self, student_performance, goal):
        """
        Generates a student performance diagnosis and a personalized study plan.
        
        Args:
            student_performance (dict): A dictionary containing student performance metrics.
            goal (str): The student's academic or career goal.
            
        Returns:
            str: The LLM's response containing diagnosis and study plan.
        """
        
        # Identify weak areas using the diagnosis utility
        weak_areas = self.diagnosis_tool.identify_weak_areas(student_performance)
        weak_areas_text = self.diagnosis_tool.format_weak_areas_for_prompt(weak_areas)
        
        system_prompt = """
        You are an expert academic counselor. Your task is to provide a detailed diagnosis of a student's performance, 
        specifically focusing on their weak areas, and create a structured study plan to help them reach their goals.
        
        Format your response with the following sections:
        1. **Performance Diagnosis**: A critical analysis of where the student stands.
        2. **Weak Areas**: Detailed breakdown of subjects with low performance.
        3. **Strengths**: Acknowledging subjects where the student is excelling.
        4. **Personalized Study Plan**: An actionable, week-by-week plan to achieve their goal.
        5. **Recommended Resources**: Targeted resources for their weakest subjects.
        """
        
        user_prompt = f"""
        **Student Performance Data:**
        {student_performance}
        
        **Identified Weak Areas:**
        {weak_areas_text}
        
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