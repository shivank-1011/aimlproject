import pandas as pd
import numpy as np

class StudentDiagnosis:
    def __init__(self, critical_threshold=60, improvement_threshold=75):
        self.critical_threshold = critical_threshold
        self.improvement_threshold = improvement_threshold
        self.subjects = ['Maths', 'SESD', 'AIML', 'FSD', 'DVA']
        self.subject_descriptions = {
            'Maths': 'Mathematical reasoning and problem solving',
            'SESD': 'Software Engineering and System Design',
            'AIML': 'Artificial Intelligence and Machine Learning',
            'FSD': 'Full Stack Development',
            'DVA': 'Data Visualization and Analytics'
        }

    def identify_weak_areas(self, student_data):
        """
        Identifies critical and improvement areas based on scores.
        
        Args:
            student_data (dict): Dictionary with subject names as keys and scores as values.
            
        Returns:
            dict: {
                "critical": list of dicts for scores < critical_threshold,
                "improvement": list of dicts for scores between thresholds,
                "strengths": list of dicts for scores > improvement_threshold
            }
        """
        diagnosis = {
            "critical": [],
            "improvement": [],
            "strengths": []
        }
        
        scores_list = []
        for sub in self.subjects:
            if sub in student_data:
                try:
                    score = float(student_data[sub])
                    entry = {
                        "subject": sub,
                        "score": score,
                        "description": self.subject_descriptions.get(sub, "")
                    }
                    scores_list.append(entry)
                    
                    if score < self.critical_threshold:
                        entry["reason"] = f"{sub}: Critical performance gap"
                        diagnosis["critical"].append(entry)
                    elif score < self.improvement_threshold:
                        entry["reason"] = f"{sub}: Needs attention to improve"
                        diagnosis["improvement"].append(entry)
                    else:
                        entry["reason"] = f"{sub}: Strong performance"
                        diagnosis["strengths"].append(entry)
                except (ValueError, TypeError):
                    continue
        
        # If no critical or improvement areas, but still have subjects, 
        # mark at least the lowest score as an improvement area if it's not a strength
        if not diagnosis["critical"] and not diagnosis["improvement"] and scores_list:
            scores_list.sort(key=lambda x: x['score'])
            lowest = scores_list[0]
            if lowest["score"] < 85: # Even if it's above 75, we can highlight the lowest if it's below an 'A' grade
                lowest["reason"] = f"{lowest['subject']}: Lowest scoring area (Opportunity for excellence)"
                diagnosis["improvement"].append(lowest)

        # Sort within each list
        for key in diagnosis:
            diagnosis[key].sort(key=lambda x: x['score'])
            
        return diagnosis

    def format_weak_areas_for_prompt(self, diagnosis):
        """Formats the diagnosis into a clear text for LLM consumption."""
        prompt_lines = []
        
        if diagnosis["critical"]:
            prompt_lines.append("### Critical Areas (Urgent):")
            for area in diagnosis["critical"]:
                prompt_lines.append(f"- {area['reason']} (Score: {area['score']})")
        
        if diagnosis["improvement"]:
            prompt_lines.append("\n### Areas for Improvement:")
            for area in diagnosis["improvement"]:
                prompt_lines.append(f"- {area['reason']} (Score: {area['score']})")
                
        if not prompt_lines:
            return "General academic excellence detected across all core subjects."
        
        return "\n".join(prompt_lines)

if __name__ == "__main__":
    # Test logic
    tester = StudentDiagnosis()
    sample = {'Maths': 68, 'SESD': 82, 'AIML': 90, 'FSD': 70, 'DVA': 88}
    diag = tester.identify_weak_areas(sample)
    print(tester.format_weak_areas_for_prompt(diag))