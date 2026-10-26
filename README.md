A simple worker-judge agent for local AI on Linux.

### Installation
clone project, then from project root run `uv tool install .`. Naturally, you must have uv installed for this.

### Usage example
`MODEL="model_name" BASE_URL="http://localhost:8080/v1" TEMPERATURE=1.0 agent "$(cat TASK.md)"`


