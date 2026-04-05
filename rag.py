import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class StudyResourceRAG:
    def __init__(self, resource_file='resources.json', model_name='all-MiniLM-L6-v2'):
        """
        Initializes the RAG system by loading and indexing study resources.
        """
        self.resource_file = resource_file
        self.model = SentenceTransformer(model_name)
        
        # Load resources
        if os.path.exists(resource_file):
            with open(resource_file, 'r') as f:
                self.resources = json.load(f)
        else:
            self.resources = []
            print(f"Warning: Resource file {resource_file} not found.")

        # Create Index
        if self.resources:
            self._build_index()
        else:
            self.index = None

    def _build_index(self):
        """
        Embeds the topics and builds a FAISS index for similarity search.
        """
        # We combine topic and subject for better matching context
        self.texts = [f"{r['topic']} in {r['subject']}" for r in self.resources]
        
        embeddings = self.model.encode(self.texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        # print("FAISS index built successfully.")

    def retrieve_resources(self, query, top_k=2):
        """
        Retrieves top_k resources that match the query.
        
        Args:
            query (str): The search query (e.g., 'Weak area in Maths').
            top_k (int): Number of resources to retrieve.
            
        Returns:
            list: A list of dicts containing the top matching resources.
        """
        if not self.index or not self.resources:
            return []

        # Embed query
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.resources) and idx != -1:
                results.append(self.resources[idx])
                
        return results

    def get_resources_for_diagnosis(self, weak_areas):
        """
        Takes a list of weak areas and retrieves relevant resources for each.
        
        Args:
            weak_areas (list): List of dicts (from StudentDiagnosis.identify_weak_areas).
            
        Returns:
            dict: {subject: [resources]}
        """
        recommendations = {}
        for area in weak_areas:
            subject = area['subject']
            reason = area['reason']
            query = f"{subject} {reason} related topics"
            
            recommendations[subject] = self.retrieve_resources(query)
            
        return recommendations

if __name__ == "__main__":
    # Test session
    rag = StudyResourceRAG()
    
    # Test single query
    print("--- Searching for: Linear Algebra ---")
    results = rag.retrieve_resources("Maths Linear Algebra and Probability")
    for r in results:
        print(f"Topic: {r['topic']} | Subject: {r['subject']} | Link: {r['link']}")
    
    # Test with typical weak area input
    print("\n--- Resources for Weak Areas ---")
    sample_weak_areas = [
        {"subject": "AIML", "reason": "AIML: Critical performance gap", "score": 32},
        {"subject": "FSD", "reason": "FSD: Needs attention to improve", "score": 70}
    ]
    diag_recs = rag.get_resources_for_diagnosis(sample_weak_areas)
    for sub, recs in diag_recs.items():
        print(f"Recommendations for {sub}:")
        for r in recs:
            print(f"  - {r['topic']} ({r['link']})")