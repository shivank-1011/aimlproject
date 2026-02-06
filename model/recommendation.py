import pandas as pd
import os

def generate_recommendations():
    INPUT_PATH = 'data/predictions/classification_report.csv'
    OUTPUT_DIR = 'data/predictions'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'student_recommendations.csv')

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        print("Please run 'python model/predict.py' first")
        return

    print(f"Loading predictions from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'Category' not in df.columns:
        print("Error: 'Category' column missing in input file.")
        return

    recommendations_map = {
        'At-risk': [
            "Focus on fundamental concepts and weak topics.",
            "Increase daily practice to at least 2 hours.",
            "Seek immediate mentorship or peer support."
        ],
        'Average': [
            "Systematically revise identified weak areas.",
            "Take weekly mock tests to improve exam temperament.",
            "Analyze errors in depth to prevent recurrence."
        ],
        'High-performing': [
            "Solve advanced and competitive-level problems.",
            "Practice under strict time constraints.",
            "Optimize solutions for better time/space complexity."
        ]
    }

    print(" Applying recommendation rules...")
    
    def get_advice(category):
        advice_list = recommendations_map.get(category, ["No specific recommendation available."])
        return " | ".join(advice_list)

    df['Study_Advice'] = df['Category'].apply(get_advice)

    print(f"Saving recommendations to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Recommendation Sample ---")
    for category, advice in recommendations_map.items():
        print(f"\n[{category}]")
        for tip in advice:
            print(f" - {tip}")

    print(f"\nSuccessfully generated recommendations for {len(df)} students.")

if __name__ == "__main__":
    generate_recommendations()