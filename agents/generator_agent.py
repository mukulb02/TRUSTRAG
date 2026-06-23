from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

GENERATOR_MODEL = "facebook/bart-large-cnn"


class GeneratorAgent:
    """
    BART-large-CNN based answer generator.

    BART is a summarisation model — it works best when given
    a passage of text to summarise rather than instruction-following prompts.
    We format the input as a document with the evidence and question,
    then strip any prompt leakage from the output.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GeneratorAgent] Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        self.model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(GENERATOR_MODEL)
            .to(self.device)
        )
        self.model.eval()
        print("[GeneratorAgent] Ready.")

    # --------------------------------------------------
    # Build context from retrieved docs
    # --------------------------------------------------

    def build_context(self, docs: list, max_docs: int = 5) -> str:
        """Extract clean answer text from each retrieved Q&A doc."""
        import re
        parts = []
        for doc in docs[:max_docs]:
            text = doc["text"]
            # Extract only the assistant answer — cleaner for summarisation
            match = re.search(r'<ASSISTANT>:\s*(.*)', text, re.DOTALL)
            if match:
                parts.append(match.group(1).strip())
            else:
                parts.append(re.sub(r'<HUMAN>:|<ASSISTANT>:', '', text).strip())
        return "\n\n".join(parts)

    # --------------------------------------------------
    # Build prompt suited for BART summarisation
    # BART works best with passage-style input, not instructions
    # --------------------------------------------------

    def build_prompt(self, query: str, context: str) -> str:
        """
        Format as a document for BART to summarise.
        Avoids instruction-style text that leaks into BART output.
        """
        return (
            f"Question: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Answer:"
        )

    # --------------------------------------------------
    # Clean output — strip prompt leakage
    # --------------------------------------------------

    def clean_output(self, text: str) -> str:
        """
        Remove common BART prompt leakage patterns.
        BART sometimes copies parts of the input into the output.
        """
        # Remove instruction phrases that leak through
        patterns = [
            r"You are a medical question-answering assistant\..*?clearly\.",
            r"Trust Score:.*?%\.",
            r"Use only the evidence below.*?clearly\.",
            r"^Answer:\s*",
            r"^Evidence:\s*",
            r"^Question:.*?\n",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text if text else "Insufficient evidence to provide a reliable answer."

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    def generate(
        self,
        query: str,
        docs: list,
        trust_score: float
    ) -> str:

        context = self.build_context(docs)
        prompt  = self.build_prompt(query, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,           # beam search — more coherent than sampling
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.clean_output(raw)