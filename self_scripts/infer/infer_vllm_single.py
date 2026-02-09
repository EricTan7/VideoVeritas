import base64
from openai import OpenAI
import argparse


## Input prompts
SYSTEM_PROMPT = """You are an expert video analyst.
Please think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Finally, give a final verdict within <answer> </answer> tags."""
USER_PROMPT = """Is this video real or fake?"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')   
    parser.add_argument('--model', type=str,
                        default='model', help='model type')
    parser.add_argument('--video_path', type=str,
                        default='', help='model type')
    args = parser.parse_args()
    
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"

    vllm_server_config = {
        "model": {
            "api": "http://localhost:8000/v1",
            "model": "/path/to/VideoVeritas"
        }
    }
    test_model = args.model
    config = vllm_server_config[test_model]

    openai_api_base = config["api"]
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    video_path = args.video_path
    with open(video_path, "rb") as f:
        image_encoded = base64.b64encode(f.read()).decode("utf-8")
    base64_image = f'data:video/mp4;base64,{image_encoded}'
    message_mm = {"type": "video_url", "video_url": {"url": base64_image}}


    try:
        chat_response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        message_mm,
                        {"type": "text", "text": USER_PROMPT}
                    ],
                },
            ],
            temperature=0.7,
            extra_body={"mm_processor_kwargs": {"fps": 3, "do_sample_frames": True}}
        )
        full_content = chat_response.choices[0].message.content
        print(video_path)
        print(full_content.strip())
    except:
        print("failed: ", video_path)
