import gradio as gr
from .nlp_processor import NLPProcessor
from .rishi_personality import RishiPersonality

class GradioUIDynamic:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.personality = RishiPersonality(tone="gentle")

    def parse_intent(self, query: str) -> dict:
        entities = self.nlp.extract_entities(query)
        return {
            "chakra": entities.get("chakra")[0] if entities.get("chakra") else "heart",
            "mantra": entities.get("mantra")[0] if entities.get("mantra") else "Om Shanti",
            "emotion": entities.get("emotion")[0] if entities.get("emotion") else "peace",
            "goal": entities.get("goal")[0] if entities.get("goal") else "healing"
        }

    def process_query(self, query: str) -> str:
        context = self.parse_intent(query)
        return self.personality.respond(query, context)

    def launch_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Divyam Rishi: Whispering Vedas")
            query = gr.Textbox(label="Your Spiritual Query")
            output = gr.Textbox(label="Guidance")
            submit = gr.Button("Seek Wisdom")
            submit.click(self.process_query, inputs=query, outputs=output)
        demo.launch(server_name="localhost", server_port=7860)
