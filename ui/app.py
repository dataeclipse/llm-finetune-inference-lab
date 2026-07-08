import argparse
from typing import Any

from lab.data.format import SYSTEM_PROMPT
from lab.serving.client import OpenAICompatClient


def build_demo(base_url: str, model: str) -> Any:
    import gradio as gr

    client = OpenAICompatClient(base_url=base_url, model=model)

    async def respond(
        message: str, history: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], str, str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)
        messages.append({"role": "user", "content": message})
        result = await client.complete(messages, max_tokens=512)
        tokens_per_second = (
            result.completion_tokens / result.latency_seconds if result.latency_seconds else 0.0
        )
        stats = (
            f"latency: {result.latency_seconds:.2f}s · "
            f"tokens: {result.completion_tokens} · "
            f"speed: {tokens_per_second:.1f} tok/s"
        )
        updated = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.text},
        ]
        return updated, "", stats

    with gr.Blocks(title="text-to-SQL lab") as demo:
        gr.Markdown(f"## text-to-SQL assistant\nServing `{model}` from `{base_url}`")
        chatbot = gr.Chatbot(type="messages", height=480)
        stats_box = gr.Markdown("latency: – · tokens: – · speed: –")
        prompt_box = gr.Textbox(
            placeholder="Paste a schema and ask a question...", show_label=False
        )
        prompt_box.submit(
            respond, inputs=[prompt_box, chatbot], outputs=[chatbot, prompt_box, stats_box]
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3-8b-sql")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_demo(args.base_url, args.model)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
