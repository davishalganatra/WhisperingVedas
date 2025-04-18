import gradio as gr
from .nlp_processor import NLPProcessor
from .chakra_energy_kpi import ChakraEnergyKPI
from typing import Dict

class GradioUIDynamic:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.chakra_kpi = ChakraEnergyKPI()

    def parse_intent(self, query: str) -> Dict:
        entities = self.nlp.extract_entities(query)
        intent = "suggest_mantra"
        if "blocked" in query.lower() or "help" in query.lower():
            intent = "heal_chakra"
        elif "clarity" in query.lower() or "focus" in query.lower():
            intent = "suggest_mantra"
        elif "tapasya" in query.lower():
            intent = "start_tapasya"
        
        return {
            "intent": intent,
            "emotion": entities.get("emotion")[0] if entities.get("emotion") else "peace",
            "chakra": entities.get("chakra")[0] if entities.get("chakra") else "heart",
            "goal": entities.get("goal")[0] if entities.get("goal") else "healing"
        }

    def process_query(self, query: str) -> str:
        intent_data = self.parse_intent(query)
        if intent_data["intent"] == "heal_chakra":
            kpi = self.chakra_kpi.analyze(query)
            return f"Aligning {kpi['chakra']} chakra with {kpi['mantra']}. Vibrational score: {kpi['vibrational_score']}"
        elif intent_data["intent"] == "suggest_mantra":
            kpi = self.chakra_kpi.analyze(query)
            return f"For {intent_data['goal']}, chant {kpi['mantra']} to align {kpi['chakra']} chakra."
        elif intent_data["intent"] == "start_tapasya":
            return f"Initiating tapasya with focus on {intent_data['chakra']} chakra and {intent_data['emotion']}."
        return "I understand your intent. Please clarify your spiritual goal."

    def launch_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Divyam Rishi: Whispering Vedas")
            query = gr.Textbox(label="Share your spiritual query")
            output = gr.Textbox(label="Guidance")
            query.submit(self.process_query, inputs=query, outputs=output)
        demo.launch(server_name="localhost", server_port=7860)