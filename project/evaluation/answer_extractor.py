import json
import re
from openai import OpenAI
from typing import Optional, Dict, Any
from evaluation.parsing import ParsingHelper, Strategies

class AnswerExtractionPipeline:
    def __init__(
        self,
        model_name: str,
        llm_endpoint: Optional[str] = None,
        api_mode: str = "chat",
    ):
        """
        Initialize the answer extraction pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
        """
        self.model_name = model_name
        self.api_mode = api_mode

        assert llm_endpoint is not None, "LLM endpoint must be provided for answer extraction."
        self.llm = OpenAI(
            base_url=f"http://{llm_endpoint}/v1",
            api_key="EMPTY"
        )
        
    def apply_chat_template(self, model_answer: str) -> str:
        """
        Apply chat template to format the extraction prompt.
        
        Args:
            model_answer: Model's answer to parse
            
        Returns:
            Formatted prompt with chat template applied
        """
        messages = [
            {
                "role": "user",
                "content": f"""You are a helpful assistant that extracts concise answers from text. Extract only the direct answer, removing explanations.
/no_think

Given the following answer, extract ONLY the final answer in a concise format.
Do not reason. Do not explain. Do not repeat the question. Do not include words like "answer".
Return only the extracted answer, for example: 2 or A or \\frac{{1}}{{3}}.

                            Model Answer: {model_answer}

                            Extract the answer (just the answer itself, no explanations):"""
            }
        ]
        
        # # Apply the model's chat template
        # prompt = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False
        # )
        
        return messages

    def build_completion_prompt(self, model_answer: str) -> str:
        return f"""You are a helpful assistant that extracts concise answers from text.
Extract only the direct final answer, removing explanations.
/no_think

Model Answer:
{model_answer}

Extract the answer. Write only the answer itself, with no explanation:"""

    def postprocess_extracted_answer(self, extracted_answer: str) -> str:
        text = (extracted_answer or "").strip()
        if not text:
            return ""

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"^```(?:\w+)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

        # Qwen thinking models may emit "... /think 2" even when asked not to.
        if "/think" in text.lower():
            text = re.split(r"/think", text, flags=re.IGNORECASE)[-1].strip()

        for strategy in (Strategies.boxed, Strategies.xml, Strategies.keyphrase):
            parsed = strategy(text)
            if parsed:
                return ParsingHelper.clean(parsed)

        cleaned = ParsingHelper.clean(text)
        if not cleaned:
            return ""

        # If the extractor still produced a sentence, prefer the final short answer-like token.
        tokens = cleaned.split()
        if len(tokens) > 1:
            answerish_tokens = [
                token for token in tokens
                if re.fullmatch(r"[a-z]", token) or re.fullmatch(r"-?\d+(?:\.\d+)?", token)
            ]
            if answerish_tokens:
                return answerish_tokens[-1]

        return cleaned
    
    def extract_answer(
        self,
        model_answer: str,
    ) -> Dict[str, Any]:
        """
        Extract answer using vLLM inference.
        
        Args:
            model_answer: Model answer text to parse
            
        Returns:
            Extracted answer string
        """
        try:
            if self.api_mode == "chat":
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=self.apply_chat_template(model_answer),
                    temperature=0.0,
                    max_tokens=32,
                    stop=["\n\n", "Model Answer:"],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                extracted_answer = response.choices[0].message.content.strip()
            elif self.api_mode == "completion":
                response = self.llm.completions.create(
                    model=self.model_name,
                    prompt=self.build_completion_prompt(model_answer),
                    temperature=0.0,
                    max_tokens=32,
                    stop=["\n\n", "Model Answer:"],
                )
                extracted_answer = response.choices[0].text.strip()
            else:
                raise ValueError(f"Unknown answer extraction API mode: {self.api_mode}")
        except Exception as exc:
            raise RuntimeError(
                f"Answer extraction failed for model {self.model_name} at {self.llm.base_url}"
            ) from exc
        
        return self.postprocess_extracted_answer(extracted_answer)
    
def process_document(input_file, pipeline: AnswerExtractionPipeline, logger: Optional[Any] = None) -> list[dict[str, Any]]:
    """
    Process a document containing model answers, extracting answers using the pipeline.
    
    Args:
        input_file: Path to the input JSONL file with model answers
        pipeline: Initialized AnswerExtractionPipeline instance
    Returns:
        List of records with extracted answers
    """
    correct_file = []
    
    with open(input_file, "r") as f:
        for idx, line in enumerate(f):
            if logger: logger.info(f"Processing example {idx+1}")
            record = json.loads(line)
            model_answer = record["model_output"]

            result = pipeline.extract_answer(model_answer)
            record["model_parsed_answer"] = result
            correct_file.append(record)

    with open(input_file, "w") as f:
        for record in correct_file:
            f.write(json.dumps(record) + "\n")

    return correct_file

   


    
