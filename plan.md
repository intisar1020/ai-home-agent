gpt2-toolcall-dataset/
│
├── README.md
├── requirements.txt
├── config.py
│
├── generate.py              # Main entry point
├── ollama_client.py         # Ollama wrapper
├── generator.py             # Dataset generation pipeline
├── validator.py             # Pydantic validation
├── prompts.py               # Prompt templates
├── tools.py                 # Tool definitions
├── schemas.py               # Pydantic schemas
├── utils.py                 # Helpers
│
├── output/
│      dataset.jsonl
│      rejected.jsonl
│
└── logs/
       generation.log