# MCP Server - Whispering Vedas 
 
 
## Features 
- **Mantra Suggestion**: Suggests mantras based on emotion, goal, and chakra. 
- **Chant Generation**: Generates chant audio via an external webhook. 
- **Event Logging**: Stores user events in local JSON logs. 
- **Tapasya Scheduling**: Schedules mantra practices using APScheduler. 
- **Voice Feedback**: Placeholder for future audio analysis. 
- **Extensibility**: Modular design for plugins (mobile UI, LLMs, dashboards). 
 
## Prerequisites 
- Python 3.9+ 
- A local chant generation service running at `http://localhost:8081/generate` (or mock it for testing). 
- Local storage for logs (`data/user_logs/`). 
 
## Setup 
1. Clone the repository: 
   ```bash 
