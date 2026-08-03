import re

class ParsingHelper:
    """Shared cleaning logic (internal use)"""
    @staticmethod
    def extract_all_boxed(text):
        results = []
        start_marker = "boxed{" 
        current_idx = 0
        while True:
            start_idx = text.find(start_marker, current_idx)
            if start_idx == -1: break
            scan_idx = start_idx + len(start_marker)
            brace_depth = 1
            content_start = scan_idx
            for i in range(scan_idx, len(text)):
                if text[i] == '{': brace_depth += 1
                elif text[i] == '}': brace_depth -= 1
                if brace_depth == 0:
                    results.append(text[content_start:i])
                    current_idx = i
                    break
            else: break
        return results

    @staticmethod
    def extract_last_boxed(text: str) -> str:
        boxed = ParsingHelper.extract_all_boxed(text)
        return boxed[-1] if boxed else ""

    @staticmethod
    def clean(text: str) -> str:

        text = str(text)

        if not text: 
            return ""

        # 1. Strip LaTeX commands (e.g. \text, \$, \frac)
        #    This turns "\text{quarter past}" into "{quarter past}"
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # 2. Lowercase
        text = text.lower()

        # 3. ALLOW SPACES: Add '\s' to the allowed list
        #    We keep: a-z, 0-9, dots, minus, slashes, AND whitespace
        text = re.sub(r"[^a-z0-9\.\-\/\s]", "", text)

        # 4. Normalize Whitespace
        #    Convert "  quarter   past  " -> "quarter past"
        text = re.sub(r"\s+", " ", text).strip()
        
        # 5. Remove boxed
        if "boxed" in text:
            text = text.split("boxed{")[-1]
            text = text[:-1] if text.endswith("}") else text

        if "answer" in text:
            # answerthe answer is 
            text = text.replace("the answer is", "").replace("/answer", "").replace("answer", "")

        if len(text) == 2 and text[1] == ".":
            text = text[0]

        if len(text) >= 3 and text[1] == "." and text[2] == " ":
            text = text[0]

        return text



class Strategies:  
    
    @staticmethod
    def boxed(text: str) -> str:
        """Strictly looks for \boxed{}. Returns the LAST one found."""
        boxes = ParsingHelper.extract_all_boxed(text)
        return boxes[-1] if boxes else ""

    @staticmethod
    def xml(text: str) -> str:
        """Strictly looks for <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def keyphrase(text: str) -> str:
        """
        Heuristic-based extraction. 
        Iterates through a list of common "Answer: X" and sentence patterns.
        """
        # Patterns to check in order (from most explicit to least explicit)
        patterns = [
            # 1. Explicit Labels: "Answer: 5", "Final value: 0.0", "Result: A"
            r"(?:answer|final answer|final value|result)[:\s]+([^\n\.]+)(?:\.|$|\n)",
            
            # 2. Natural phrasing: "The answer is 5", "The result is 10"
            r"(?:answer is|result is|value is)[:\s]+([^\n\.]+)(?:\.|$|\n)",
            
            # 3. Conclusion Sentences (Common in MCQs):
            # Matches: "The correct option is C", "The correct answer is (A)"
            r"correct option is[:\s]+([a-eA-E0-9\s]+)(?:\.|_|\)|\s|$)",
            r"correct answer is[:\s]+(?:\()([a-eA-E0-9\s]+)(?:\))", # Catches (A)
            r"correct answer is[:\s]+([^\n\.]+)(?:\.|$)"
        ]

        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def loose_number(text: str) -> str:
        """Last resort: Grabs the last number found in text."""
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else ""