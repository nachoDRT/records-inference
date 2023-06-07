import json
import os
import pandas as pd
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch

sys.path.append(
    os.path.join(
        str(Path(os.path.abspath(__file__)).parent.parent),
        "9_Transformers_Git",
        "transformers",
        "src",
    )
)

from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import (
    LayoutLMv2FeatureExtractor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AdamW,
    TrainingArguments,
    Trainer,
)
from pathlib import Path

SHOW_IMG = False
NUM_SAMPLES = 25
TOTAL_STEPS = 5000
SAVE_STEPS = 1000
F_TO_EVALUATE = 2


def load_dataset_from_json():
    json_dataset_file_path = os.path.join(
        os.path.dirname(__file__), "docVQA", "train", "train_v1.0.json"
    )

    with open(json_dataset_file_path) as f:
        data = json.load(f)

    return data


def load_dataset():
    data = load_dataset_from_json()
    df = pd.DataFrame(data["data"])
    dataset = Dataset.from_pandas(df.iloc[:NUM_SAMPLES])

    return dataset


def get_ocr_words_and_boxes(samples):
    images = [
        Image.open(
            os.path.join(os.path.dirname(__file__), "docVQA", "train", image_file)
        ).convert("RGB")
        for image_file in dataset["image"]
    ]

    encoded_inputs = feature_extractor(images)

    samples["image"] = encoded_inputs.pixel_values
    samples["words"] = encoded_inputs.words
    samples["boxes"] = encoded_inputs.boxes
    samples["image_name"] = dataset["image"]

    return samples


def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if (
            words_list[i] == answer_list[0]
            and words_list[i : i + len(answer_list)] == answer_list
        ):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    # XXX Apend para luego devolver la primera posición de la lista. ¿Tiene sentido?
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def show_image(img, name: str):
    reshaped_image = np.transpose(img, (1, 2, 0))
    reshaped_image = reshaped_image.astype(np.uint8)
    cv2.imshow(name, reshaped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def encode_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

    questions = dataset["question"]
    words = dataset["words"]
    boxes = dataset["boxes"]
    answers = dataset["answers"]
    images = dataset["image"]

    encoding = tokenizer(
        questions, words, boxes, max_length=512, padding="max_length", truncation=True
    )

    start_positions = []
    end_positions = []

    for batch_index in range(len(answers)):
        print("")
        # print("Palabras: \n", words[batch_index])
        print("Pregunta: \n", questions[batch_index])
        print("Respuesta: \n", answers[batch_index])

        if SHOW_IMG:
            img = np.array(images[batch_index])
            show_image(img, dataset["image_name"][batch_index])

        # Get the index position of the CLS token in the input_ids
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        words_dataset = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(
                words_dataset, answer.lower().split()
            )
            if match:
                break

        if match:
            sequence_ids = encoding.sequence_ids(batch_index)
            # Get the index position of the first word of the answer in the sequence_ids
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # Get the index position of the last word of the answer in the sequence_ids
            token_end_index = len(sequence_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            word_ids = encoding.word_ids(batch_index)[
                token_start_index : token_end_index + 1
            ]
            # print(encoding.word_ids(batch_index))
            # print(word_ids)
            for id in word_ids:
                if id == word_idx_start:
                    start_positions.append(token_start_index)
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions.append(token_end_index)
                    break
                else:
                    token_end_index -= 1

            print("Verifying start position and end position:")
            print("True answer:", answer)
            start_position = start_positions[batch_index]
            end_position = end_positions[batch_index]
            reconstructed_answer = tokenizer.decode(
                encoding.input_ids[batch_index][start_position : end_position + 1]
            )
            print("Reconstructed answer:", reconstructed_answer)
            print("-----------")

        else:
            print("No match found for answer:", answer)
            print("-----------")
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding["image"] = dataset["image"]
    encoding["start_positions"] = start_positions
    encoding["end_positions"] = end_positions

    return encoding


def define_features():
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "image": Array3D(dtype="int64", shape=(3, 224, 224)),
            "start_positions": Value(dtype="int64"),
            "end_positions": Value(dtype="int64"),
        }
    )

    return features


def create_data_loaders(dataloader):
    encoded_dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=2)

    return dataloader


if __name__ == "__main__":
    dataset = load_dataset()
    feature_extractor = LayoutLMv2FeatureExtractor()
    dataset_with_ocr = dataset.map(
        get_ocr_words_and_boxes, batched=True, batch_size=NUM_SAMPLES
    )
    # encoding = encode_dataset(dataset_with_ocr)

    features = define_features()
    encoded_dataset = dataset_with_ocr.map(
        encode_dataset,
        batched=True,
        batch_size=NUM_SAMPLES,
        remove_columns=dataset_with_ocr.column_names,
        features=features,
    )

    # dataloader = create_data_loaders(encoded_dataset)

    model_checkpoint = "microsoft/layoutlmv2-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    args = TrainingArguments(
        output_dir="qa_layoutxlm-finetuned-es_trunds",  # name of directory to store the checkpoints
        overwrite_output_dir=True,
        max_steps=TOTAL_STEPS,  # we train for a maximum of 1,000 batches
        warmup_ratio=0.1,  # we warmup a bit
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        push_to_hub=False,  # we'd like to push our model to the hub during training
        save_steps=SAVE_STEPS,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset,
    )

    trainer.train(frequency_to_evaluate=F_TO_EVALUATE)

    # model.train()

    # for epoch in range(20):  # loop over the dataset multiple times
    #     for idx, batch in enumerate(dataloader):
    #         # get the inputs;
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         token_type_ids = batch["token_type_ids"].to(device)
    #         bbox = batch["bbox"].to(device)
    #         image = batch["image"].to(device)
    #         start_positions = batch["start_positions"].to(device)
    #         end_positions = batch["end_positions"].to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             token_type_ids=token_type_ids,
    #             bbox=bbox,
    #             image=image,
    #             start_positions=start_positions,
    #             end_positions=end_positions,
    #         )
    #         loss = outputs.loss
    #         print("Loss:", loss.item())
    #         loss.backward()
    #         optimizer.step()
