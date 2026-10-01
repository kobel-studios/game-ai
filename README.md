# GameAI

A local experimental game-learning agent that watches the screen, records player input, learns from feedback, and can optionally control a game. It supports screen capture, imitation data, reward signals, reusable routines, and local Ollama vision/text models.

## Requirements

- Windows
- Python 3.10+
- Optional: [Ollama](https://ollama.com/) with the models configured in the selected profile

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python gameai.py --profile helldivers2 --gui
```

Use watch mode first. Control mode generates keyboard and mouse input; supervise it and use the configured pause/quit hotkeys. Captured frames, learned models, and episode data remain local and are excluded from Git.
