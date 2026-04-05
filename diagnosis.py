import pandas as pd
import numpy as np

class StudentDiagnosis:
    def __init__(self, critical_threshold=60, improvement_threshold=75, subjects=None):
        self.critical_threshold = critical_threshold
        self.improvement_threshold = improvement_threshold
        # If subjects are not provided, we detect them dynamically in identify_weak_areas
        self.subjects = subjects
        self.subject_descriptions = {
            'Maths': 'Mathematical reasoning and problem solving',
            'SESD': 'Software Engineering and System Design',
            'AIML': 'Artificial Intelligence and Machine Learning',
            'FSD': 'Full Stack Development',
            'DVA': 'Data Visualization and Analytics'
        }

    def _get_dynamic_subjects(self, student_data):
        """
        Dynamically filters subjects from student data.
        Recognizes numeric columns that look like grades (0-100).
        """
        detected = []
        exclude_keywords = ['id', 'score', 'status', 'prob', 'total', 'attendance', 'name', 'urn', 'section']
        
        for key, value in student_data.items():
            # Check if key is not in excluded names
            if any(word in key.lower() for word in exclude_keywords):
                continue
                
            try:
                # Check if the value is a number between 0 and 100
                score = float(value)
                if 0 <= score <= 100:
                    detected.append(key)
            except (ValueError, TypeError):
                continue
        
        return detected

    def identify_weak_areas(self, student_data):
        """
        Identifies critical and improvement areas based on real data scores.
        """
        diagnosis = {
            "critical": [],
            "improvement": [],
            "strengths": []
        }
        
        # Use provided subjects or detect them dynamically
        subjects_to_check = self.subjects if self.subjects else self._get_dynamic_subjects(student_data)
        
        scores_list = []
        for sub in subjects_to_check:
            if sub in student_data:
                try:
                    score = float(student_data[sub])
                    entry = {
                        "subject": sub,
                        "score": score,
                        "description": self.subject_descriptions.get(sub, f"Study material for {sub}")
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
        
        # If no critical or improvement areas, but still have subjects, identify the relative weakest
        if not diagnosis["critical"] and not diagnosis["improvement"] and scores_list:
            scores_list.sort(key=lambda x: x['score'])
            lowest = scores_list[0]
            # If the lowest is still less than an 85, flag it for excellence
            if lowest["score"] < 85:
                lowest["reason"] = f"{lowest['subject']}: Lowest scoring area (Opportunity for excellence)"
                diagnosis["improvement"].append(lowest)

        # Sort results
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
            return "General academic excellence detected across all subjects."
        
        return "\n".join(prompt_lines)

if __name__ == "__main__":
    # Test dynamic detection
    tester = StudentDiagnosis()
    sample = {
        'Advanced_Calculus': 45, 
        'Python_Programming': 82, 
        'student_name': 'Rohan',
        'attendance_percentage': 95,
        'Database_Mgmt': 58
    }
    diag = tester.identify_weak_areas(sample)
    print("--- Dynamic Diagnosis Test ---")
    print(tester.format_weak_areas_for_prompt(diag))