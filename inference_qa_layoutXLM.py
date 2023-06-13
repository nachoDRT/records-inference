import sys
import os
from pathlib import Path
import torch
from PIL import Image

sys.path.append(
    os.path.join(
        str(Path(os.path.abspath(__file__)).parent.parent),
        "9_Transformers_Git",
        "transformers",
        "src",
    )
)

from transformers import (
    LayoutLMv2Processor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

docVQA = "docVQA_Grades"
path = os.path.join(
    os.path.dirname(__file__), docVQA, "train", "test", "qa_test_0001.png"
)

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(path).convert("RGB")
question = "What is the grade?"

encoding = processor(image, path, question, return_tensors="pt")
truncated_encoding = encoding[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForQuestionAnswering.from_pretrained(
    os.path.join(os.getcwd(), "qa_layoutxlm-finetuned-es_trunds")
)
model.to(device)

for key, value in truncated_encoding.items():
    truncated_encoding[key] = value.to(model.device)

outputs = model(**truncated_encoding)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

predicted_start_idx = start_logits.argmax(-1).item()
predicted_end_idx = end_logits.argmax(-1).item()
print("Predicted start idx:", predicted_start_idx)
print("Predicted end idx:", predicted_end_idx)
print(
    "\n PREDICCIÓN:",
    processor.tokenizer.decode(
        truncated_encoding.input_ids.squeeze()[
            predicted_start_idx : predicted_end_idx + 1
        ]
    ),
)
