import gradio as gr
import os
import json

class AdminConsole:
    def __init__(self, log_dir: str = "data/usage_logs", memory_dir: str = "data/memory_snapshots"):
        self.log_dir = log_dir
        self.memory_dir = memory_dir

    def view_logs(self, user_id: str) -> str:
        log_file = os.path.join(self.log_dir, f"{user_id}.json")
        if not os.path.exists(log_file):
            return "No logs found for user"
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), indent=2)

    def view_memory(self, user_id: str) -> str:
        memory_file = os.path.join(self.memory_dir, f"{user_id}_summary.json")
        if not os.path.exists(memory_file):
            return "No memory snapshot found for user"
        with open(memory_file, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), indent=2)

    def launch(self, password: str):
        if password != "sacred_rishi":
            return "Access denied"
        with gr.Blocks() as demo:
            gr.Markdown("# Divyam Rishi Admin Console")
            user_id = gr.Textbox(label="User ID")
            log_btn = gr.Button("View Logs")
            memory_btn = gr.Button("View Memory")
            output = gr.Textbox(label="Output")
            log_btn.click(self.view_logs, inputs=user_id, outputs=output)
            memory_btn.click(self.view_memory, inputs=user_id, outputs=output)
        demo.launch(server_name="localhost", server_port=7861)
