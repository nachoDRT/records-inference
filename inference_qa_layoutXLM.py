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

QUESTION = "Focus your attention on the relationship between words and digits. Here are several examples: zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10. What number corresponds to the word "

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

docVQA = "docVQA_Grades"
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

path = os.path.join(os.path.dirname(__file__), docVQA, "test")
for file in sorted(os.listdir(path)):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(path, file)
        image = Image.open(img_path).convert("RGB")
        question = "".join([QUESTION, "three?"])

        encoding = processor(image, img_path, question, return_tensors="pt")
        truncated_encoding = encoding[0]
        # print(truncated_encoding.keys())
        # print(truncated_encoding["bbox"])
        # print(Z)

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
        print(question)
        print(
            "\n PREDICCIÓN:",
            processor.tokenizer.decode(
                truncated_encoding.input_ids.squeeze()[
                    predicted_start_idx : predicted_end_idx + 1
                ]
            ),
        )
