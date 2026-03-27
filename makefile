

install:
	uv sync

generate-prompt:
	uv run python generate_prompt.py