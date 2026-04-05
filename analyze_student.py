import sys
import json
from groq_analysis import GroqAnalyzer

def main():
    """
    Command-line interface for student performance diagnosis and study plan generation.
    """
    try:
        analyzer = GroqAnalyzer()
        
        print("\n=== Student Analysis System ===")
        print("Please enter the student's performance metrics (as a JSON-like string) or individual values.")
        
        # Gathering performance data
        performance = {}
        subjects = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
        for sub in subjects:
            while True:
                try:
                    val = input(f"Enter score for {sub} (0-100) [Default 50]: ").strip()
                    if not val:
                        performance[sub] = 50
                    else:
                        performance[sub] = int(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        attendance = input("Enter overall attendance (%) [Default 75]: ").strip() or "75"
        performance['Attendance'] = f"{attendance}%"
        
        goal = input("\nEnter the student's primary academic or career goal: ").strip()
        if not goal:
            goal = "Clear all backlogs and prepare for placement."

        print("\n--- Generating Diagnosis and Study Plan ---")
        diagnosis = analyzer.generate_study_plan(performance, goal)
        
        print("\n----------------------------------------------")
        print(diagnosis)
        print("----------------------------------------------")
        
        with open("student_diagnosis_report.md", "w") as f:
            f.write(diagnosis)
        print("\nReport saved to: student_diagnosis_report.md")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Make sure you have set GROQ_API_KEY in your .env file.")
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
