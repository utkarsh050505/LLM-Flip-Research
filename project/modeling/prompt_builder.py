
from typing import Any, Optional



def build_prompt(sample: dict[str, Any], 
                 system_prompt: Optional[str] = None, 
                 reasoning_prompt: Optional[str] = None,
                 prepend_reasoning_prompt: bool = False
                ) -> list[dict]:

    """Build the prompt structure for the model based on the sample.
    
    Sample contains "images" (list of PIL Images, optionally empty) and
    "question" (already formatted with hints).
    """
    messages: list[dict[str, Any]] = []

    system_content_parts = []
    if system_prompt:
        system_content_parts.append(system_prompt)
    if reasoning_prompt and prepend_reasoning_prompt:
        system_content_parts.append(reasoning_prompt)

    # Do not send an empty system message: some chat templates treat its
    # presence as overriding the model's built-in default system prompt.
    if system_content_parts:
        messages.append(
            {
                "role": "system",
                "content": "\n".join(system_content_parts),
            }
        )

    user_text = sample["question"]
    if reasoning_prompt and not prepend_reasoning_prompt:
        user_text = f"{user_text}\n\n{reasoning_prompt}"

    if sample["images"]:
        user_content: Any = [
            *[{"type": "image", "image": "placeholder"} for _ in sample["images"]],
            {
                "type": "text",
                "text": user_text,
            },
        ]
    else:
        user_content = user_text

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    return messages
    
