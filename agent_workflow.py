import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from groq_analysis import GroqAnalyzer
from diagnosis import StudentDiagnosis
from rag import StudyResourceRAG

# Define the Agent State
class AgentState(TypedDict):
    student_data: dict
    student_goal: str
    diagnosis: dict
    weak_areas_text: str
    resources: dict
    resources_text: str
    final_report: str

class StudentAnalysisAgent:
    def __init__(self):
        self.diagnostician = StudentDiagnosis()
        self.rag_tool = StudyResourceRAG()
        self.llm_tool = GroqAnalyzer()
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        # Initialize Graph
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("diagnose", self.node_diagnose)
        workflow.add_node("plan", self.node_plan)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("generate_report", self.node_generate_report)

        # Build Edges
        workflow.set_entry_point("diagnose")
        workflow.add_edge("diagnose", "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    # Node 1: Diagnose Weak Areas
    def node_diagnose(self, state: AgentState):
        student_data = state['student_data']
        diagnosis = self.diagnostician.identify_weak_areas(student_data)
        weak_areas_text = self.diagnostician.format_weak_areas_for_prompt(diagnosis)
        return {"diagnosis": diagnosis, "weak_areas_text": weak_areas_text}

    # Node 2: Plan (Drafting logical steps)
    def node_plan(self, state: AgentState):
        # We can add planning logic or just pass through to ensure strict formatting
        return state

    # Node 3: Retrieve Targeted Resources (RAG)
    def node_retrieve(self, state: AgentState):
        diagnosis = state['diagnosis']
        all_weak = diagnosis["critical"] + diagnosis["improvement"]
        resources = self.rag_tool.get_resources_for_diagnosis(all_weak)
        
        resources_text = ""
        for sub, rec_list in resources.items():
            resources_text += f"\nFor {sub}:\n"
            for r in rec_list:
                resources_text += f"- {r['topic']}: {r['link']}\n"
        
        return {"resources": resources, "resources_text": resources_text}

    # Node 4: Generate Report (Strict Format)
    def node_generate_report(self, state: AgentState):
        student_data = state['student_data']
        student_goal = state['student_goal']
        weak_areas_text = state['weak_areas_text']
        resources_text = state['resources_text']
        
        prompt = f"""
        Generate a PREMIUM student analysis report with the following styling:
        STUDENT DATA: {student_data}
        WEAK AREAS: {weak_areas_text}
        GOAL: {student_goal}
        REAL RESOURCES: {resources_text}
        
        **STYLING RULES:** 
        - Use bold headers (Section Title). 
        - Use blockquotes (`>`) for the Learning Diagnosis section.
        - Use bullet points for all lists. 
        - Ensure the Resources section is grouped by subject in a very clean way.
        
        **STRICT OUTPUT FORMAT:**
        ### 1. Learning Diagnosis
        > (Your analysis of strengths and gaps)

        ### 2. Study Plan
        (Logical progression to solve gaps, using bold text for key milestones)

        ### 3. Weekly Goals
        - (4 specific, measurable weekly targets)

        ### 4. Curated Resources
        (The specific URLs provided above, grouped beautifully)

        ### 5. Next Steps
        - (Immediate actions for the student)
        
        Ensure your response has only these 5 numbered sections. Do not include any intro or outro text.
        """
        
        # Use our existing Groq tool but with the specific agent prompt
        report = self.llm_tool.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        
        return {"final_report": report}

    def run(self, student_data, goal):
        initial_state = {
            "student_data": student_data,
            "student_goal": goal,
            "diagnosis": {},
            "weak_areas_text": "",
            "resources": {},
            "resources_text": "",
            "final_report": ""
        }
        final_output = self.workflow.invoke(initial_state)
        return final_output['final_report']

if __name__ == "__main__":
    # Test agent
    agent = StudentAnalysisAgent()
    sample_data = {'Maths': 45, 'AIML': 32, 'student_name': 'Ayush'}
    res = agent.run(sample_data, "Get placed in a top AI company.")
    print(res)
