import os
import sys
from pathlib import Path

sys.path.append(
    os.path.join(
        str(Path(os.path.abspath(__file__)).parent.parent),
        "9_Transformers_Git",
        "transformers",
        "src",
    )
)

from transformers import AutoModelForTokenClassification, LayoutLMv2FeatureExtractor
import cv2
from PIL import Image
from transformers import LayoutXLMTokenizerFast
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import json
import copy
import re
import requests
import csv


SHOW = False
OCR_SPY = True
WRITE_CSV_OUTPUT = True

cwd_parent = Path(os.getcwd()).parent
SUBJECTS_SEMANTIC_JSON = os.path.join(
    cwd_parent,
    "7_AdHoc_Git",
    "records-dataset",
    "res",
    "subjects",
    "subjects_spanish.json",
)

OBTAIN_METRICS = True
SEND_TO_DATABASE = False


def highlight_ocr_detections(
    *, img: Image.Image, iter: int, words: list, words_bboxes: list, conf: list
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    # C2 -> good
    c2 = np.array((102, 153, 51))
    c1 = np.array((0, 0, 255))

    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, (word, bbox) in enumerate(zip(words, words_bboxes)):
        start_point = (int(bbox[0]), int(bbox[1]))
        text_start_point = (int(bbox[0]), int(bbox[1] - 10))
        end_point = (int(bbox[2]), int(bbox[3]))
        conf_i = conf[0][i] / 100

        a = (1 - conf_i) * c1 + conf_i * c2

        img = cv2.rectangle(img, start_point, end_point, color=tuple(a), thickness=1)

        img = cv2.putText(
            img, word, text_start_point, font, font_scale, tuple(a), 2, cv2.LINE_AA
        )

        # cv2.imshow("OCR", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    img_name = "".join(["ocr_detection", str(iter), ".png"])
    save_img_path = os.path.join(os.getcwd(), "ocr_detections")

    if not os.path.isdir(save_img_path):
        os.mkdir(save_img_path)

    cv2.imwrite(os.path.join(save_img_path, img_name), img=img)

    if WRITE_CSV_OUTPUT:
        
        csv_name = os.path.join(save_img_path, "".join(["ocr_detection", str(iter), ".csv"]))
        with open(csv_name, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            # Escribir cada palabra en una fila distinta
            for word_i in words:
                writer.writerow([word_i])


def get_label_2_color():
    """Link a label to a color

    Returns:
        color_dict: (dict) a dictionary with labels as keys and colors as values
    """

    # Open '.json' file with info about subjects' synonyms
    with open(SUBJECTS_SEMANTIC_JSON, "r") as subjects_json_file:
        data = subjects_json_file.read()
        subjects_semantic = json.loads(data)

    subjects_json_file.close()

    color_dict = {
        "question": (255, 102, 0),
        "answer": (102, 153, 51),
        "header": (0, 102, 255),
        # "other": (204, 0, 153),
        "other": (100, 100, 100)
    }
    answers_list = []
    subjects_list = []

    expanded_labels = []
    expanded_tags = subjects_semantic["academic_years_tags"]

    for key in subjects_semantic["subjects"]:
        for tag in expanded_tags:
            expanded_labels.append("".join([key, tag]))

    # print(expanded_labels)
    expanded_labels = get_academic_years_labels(
        labels=expanded_labels, academic_year_tags=expanded_tags
    )

    # Loop over the dictionary to extract its keys
    # for key in subjects_semantic['subjects']:
    for key in expanded_labels:
        # Add new color to dictionary
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_dict[key] = (r, g, b)

        # Add the same color for the answer (grade)
        key_answer = "".join([key, "_answer"])
        color_dict[key_answer] = (r, g, b)
        answers_list.append(key_answer)
        subjects_list.append(key)

    return color_dict, answers_list, subjects_list


def get_academic_years_labels(*, labels: list, academic_year_tags: list):
    for year in academic_year_tags:
        labels.append(year[1:])

    return labels


def get_test_image(*, path):
    """Load an image from the test dataset

    Returns:
        image: (np.array). One RGB image
    """

    image = Image.open(path)
    image = image.convert("RGB")
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

    if SHOW:
        cv2.imshow("test_image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return "other"
    return label


def clean_boxes(true_boxes, true_predictions):
    def_true_boxes = []
    def_true_predictions = []

    for i, box in enumerate(true_boxes):
        x_0 = box[0]
        y_0 = box[1]
        x_f = box[2]
        y_f = box[3]
        a = (x_f - x_0) * (y_f - y_0)

        if a > 1:
            def_true_boxes.append(true_boxes[i])
            def_true_predictions.append(true_predictions[i])

    return def_true_boxes, def_true_predictions


def generate_json(*, json_path: str, json_number: str, dict: dict):
    json_object = json.dumps(dict, indent=4)
    json_name = os.path.join(json_path, "".join(["result_", json_number, ".json"]))

    with open(json_name, "w") as outfile:
        outfile.write(json_object)

    outfile.close()


def get_line_id(line_2_bbox: dict, bbox: list):
    bbox = str(bbox)
    for item in line_2_bbox:
        if bbox in item["bboxes"]:
            return int(item["line_id"])

    return None


def unnormalize_bboxes_in_line_2_bbox(line_2_bbox: dict):
    for item in line_2_bbox:
        item["bboxes"] = [
            str(unnormalize_box(eval(bbox), width, height)) for bbox in item["bboxes"]
        ]

    return line_2_bbox


def postprocess_outputs(
    detected_subjects: list,
    grades_tags: list,
    grades: list,
    confidences: list,
    output_num: int,
    ultra: bool = False,
):
    subjects_dict = {}

    for i, grade in enumerate(grades):
        try:
            subject = detected_subjects[i]

            # Check if this subject already has a record
            if subject in subjects_dict:
                if type(subjects_dict[subject]["grade"]) == float:
                    grades_list = [subjects_dict[subject]["grade"]]
                    ocr_confidences_list = [subjects_dict[subject]["OCR_confindence"]]
                    subject_2_grade_confidence_list = [
                        subjects_dict[subject]["question_answer_confidence"]
                    ]

                else:
                    grades_list = subjects_dict[subject]["grade"]
                    ocr_confidences_list = subjects_dict[subject]["OCR_confindence"]
                    subject_2_grade_confidence_list = subjects_dict[subject][
                        "question_answer_confidence"
                    ]

                grades_list.append(grade)
                subjects_dict[subject]["grade"] = grades_list

                ocr_confidences_list.append(confidences[i])
                subjects_dict[subject]["OCR_confindence"] = ocr_confidences_list

                subject_2_grade_confidence_list.append(
                    get_subject_2_grade_confidence(grades_tags[i], expected_tag)
                )
                subjects_dict[subject][
                    "question_answer_confidence"
                ] = subject_2_grade_confidence_list

            else:
                expected_tag = "".join([subject, "_answer"])
                subjects_dict[subject] = {}
                subjects_dict[subject]["grade"] = grade
                subjects_dict[subject]["OCR_confindence"] = confidences[i]
                subjects_dict[subject][
                    "question_answer_confidence"
                ] = get_subject_2_grade_confidence(grades_tags[i], expected_tag)

        except IndexError:
            break

    save_results_path = os.path.join(os.getcwd(), "results")

    generate_json(
        json_path=save_results_path, json_number=str(output_num), dict=subjects_dict
    )
    if ultra:
        generate_json(
            json_path=save_results_path,
            json_number="".join([str(output_num), "_ultra"]),
            dict=subjects_dict,
        )

    return subjects_dict


def get_subject_2_grade_confidence(grade_tag, expected_tag):
    if grade_tag == expected_tag:
        subject_2_grade_confidence = "high_confidence"

    else:
        """
        TODO 'medium_confidence' is now based on OCR order output, but this should be
        improved and be based on geometry (minimum distnance)
        """
        subject_2_grade_confidence = "medium_confidence"

    return subject_2_grade_confidence


def open_ground_truth_json(*, path):
    # Every ground_truth file is linked to an img: obtain the img name
    img_name = os.path.basename(path)
    img_id = img_name[:-4]
    # Obtain the linked '.json' name
    json_name = "".join([img_id, ".json"])
    ground_truth_path = os.path.join(os.path.dirname(path), json_name)

    # Open the '.json' file with ground truth info about the test documents
    with open(ground_truth_path, "r") as gt_json_file:
        data = gt_json_file.read()
        ground_truth = json.loads(data)

    gt_json_file.close()

    return ground_truth


def check_ground_truth(*, grades_dict, json_path):
    ground_truth_dict = open_ground_truth_json(path=json_path)

    # Loop over the dictionary to extract its keys
    num_mistakes = 0
    num_gt_grades = 0

    for key in ground_truth_dict:
        subject = key
        gt_subject_grade = ground_truth_dict[subject]["grade"]

        # If there is just one ground truth grade (unique instance)
        if type(gt_subject_grade) == float or type(gt_subject_grade) == int:
            required_grades = 1
            num_gt_grades += required_grades
            gt_subject_grade = [gt_subject_grade]

        # Otherwise there are multiple grades for one subject
        else:
            required_grades = len(gt_subject_grade)
            num_gt_grades += required_grades

        try:
            predicted_subject_grades = grades_dict[subject]["grade"]

            if (
                type(predicted_subject_grades) == float
                or type(predicted_subject_grades) == int
            ):
                predicted_subject_grades = [predicted_subject_grades]

            mapped_grades = 0

            for predicted_subject_grade in predicted_subject_grades:
                correct_detection_flag = False

                if mapped_grades >= required_grades:
                    break

                for gt_grade in gt_subject_grade:
                    if gt_grade - predicted_subject_grade == 0:
                        try:
                            gt_subject_grade.remove(gt_grade)
                            mapped_grades += 1
                            correct_detection_flag = True

                        except IndexError:
                            correct_detection_flag = False
                            break

                if correct_detection_flag == False:
                    num_mistakes += 1

        except KeyError:
            num_mistakes += len(gt_subject_grade)

    return num_mistakes, num_gt_grades


def generate_plot(*, stats: np.array, path: str):
    plt.clf()
    plt.style.use("ggplot")
    _, bins = np.histogram(stats, bins=4, range=(0, 4))
    ax = plt.axes()
    plt.hist(stats, bins=bins, rwidth=0.5)
    ax.set_xlabel("Number of mistakes", fontsize=14)
    ax.set_ylabel("Number of documents", fontsize=14)
    ax.set_title("Mistakes distribution")
    bins = bins[:-1]
    ax.set_xticks(bins + 0.5)
    ax.set_xticklabels(bins.astype(int))
    # TODO Write '3 or more' mistakes as 'xtic' label
    plt.savefig(os.path.join(path, "performance_histogram.png"))
    plt.close()


def postprocess_docs_stats(*, stats: list):
    post_stats = []

    for stat in stats:
        if stat < 3:
            post_stats.append(stat)
        else:
            post_stats.append(3)

    return post_stats


def generate_aggregated_metrics(
    *, errors: int, total: int, path: str, ultra: bool = False
):
    ind = [0, 1]
    width = 0.35

    # Generar las barras de fallos y aciertos
    plt.style.use("ggplot")
    plt.bar(ind, [errors, total - errors], width)

    # Agregar etiquetas y título
    plt.ylabel("# of Queries")
    plt.title("Errors vs. Right Anwers")
    plt.xticks(ind, ("Errors", "Right Answers"))

    for i, v in enumerate([errors, total - errors]):
        plt.annotate(str(v), xy=(i, v), ha="center", va="bottom")

    percentage = round((total - errors) / total * 100, 2)
    plt.text(
        0.5,
        0.5,
        "Accuracy:\n{}%".format(percentage),
        ha="center",
        va="center",
        transform=plt.gcf().transFigure,
        fontsize=14,
    )

    if not ultra:
        plt.savefig(os.path.join(path, "aggregated_performance.png"))
    else:
        plt.savefig(os.path.join(path, "aggregated_ultra_performance.png"))
    plt.close()


def spawn_student():
    student = {
        "dni": "99999999D",
        "clave_comillas": "202303542",
        "nombre": "Ignacio",
        "apellido_1": "de Rodrigo",
        "apellido_2": "Tobias",
    }

    return student


def spawn_year():
    year = {"id_curso": random.randint(20, 1000), "descripcion": "1º de Bachillerato"}

    return year


def create_subject_in_database(
    *, subject_name: str, id_subject: int, id_academic_year: dict, id_curriculo: int
):
    curriculo = {
        "id_curriculo": id_curriculo,
        "id_curso": id_academic_year,
        "id_asig": id_subject,
    }

    out_curriculo = requests.post(
        "http://localhost:8000/Subjects/subjects/curriculo", json.dumps(curriculo)
    )
    print("Out_curriculo", out_curriculo)

    subject = {"id_asignatura": id_subject, "nombre_asignatura": subject_name}

    out_subject = requests.post(
        "http://localhost:8000/Subjects/subjects/add_subject", json.dumps(subject)
    )
    print("Out_subject", out_subject)


def spawn_id_curriculo():
    return random.randint(20, 2000)


def post_subject_grade(
    *, num_grade: int, subject_id: int, student_id: str, curriculo_id: int
):
    grade = {
        "id_asignatura": subject_id,
        "clave_comillas": student_id,
        "nota": num_grade,
        "id_curriculo": curriculo_id,
    }
    out_nota = requests.post("http://localhost:8000/Students/grades", json.dumps(grade))
    print("Out_nota:", out_nota)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(os.getcwd())
model.to(device)

feature_extractor = LayoutLMv2FeatureExtractor(ocr_lang="spa")

real_docs_path = os.path.join(os.getcwd(), "inference_datasets", "A_10_test_real_sanpablo")

iter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6

label2color, answers_list, available_subjects = get_label_2_color()

documents_stats = []
total_errors = 0
total_errors_ultraprocessed = 0
total_gt_grades = 0

for file in sorted(os.listdir(real_docs_path)):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(real_docs_path, file)
        img = get_test_image(path=img_path)
        # Get img to parse to OCR

        inputs = feature_extractor(
            img,
            images_paths=img_path,
            ocr_azure=True,
            return_tensors="pt",
            token_classification=True,
        )

        words = inputs["words"]
        words_bboxes = inputs["boxes"]
        width, height = img.size
        words_bboxes = [unnormalize_box(box, width, height) for box in words_bboxes[0]]
        conf = inputs["conf"]
        line_2_bbox = unnormalize_bboxes_in_line_2_bbox(inputs["line_2_bbox"])

        if OCR_SPY:
            ocr_img = copy.deepcopy(img)
            highlight_ocr_detections(
                img=ocr_img,
                iter=iter,
                words=words[0],
                words_bboxes=words_bboxes,
                conf=conf,
            )

        tokenizer = LayoutXLMTokenizerFast.from_pretrained("microsoft/layoutxlm-base")
        encoding = tokenizer(
            inputs.words,
            boxes=inputs.boxes,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            outputs = model(
                input_ids=encoding.input_ids.to(device),
                attention_mask=encoding.attention_mask.to(device),
                bbox=encoding.bbox.to(device),
                image=inputs.pixel_values.to(device),
            )

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        width, height = img.size

        is_subword = np.array(encoding.offset_mapping.squeeze().tolist())[:, 0] != 0

        true_predictions = [
            model.config.id2label[pred]
            for idx, pred in enumerate(predictions)
            if not is_subword[idx]
        ]
        true_boxes = [
            unnormalize_box(box, width, height)
            for idx, box in enumerate(token_boxes)
            if not is_subword[idx]
        ]

        true_boxes, true_predictions = clean_boxes(true_boxes, true_predictions)

        new_list = []
        indexes_to_remove = []

        for index, sublist in enumerate(true_boxes):
            if tuple(sublist) not in new_list:
                new_list.append(tuple(sublist))
            else:
                indexes_to_remove.append(index)

        for index in reversed(indexes_to_remove):
            true_predictions.pop(index)

        # Convert tuples back to lists
        true_boxes = [list(t) for t in new_list]

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

        grades_tags = []
        grades = []
        detected_subjects = []
        confidences = []
        lines_with_detected_subjects = []

        # grades_tags_ultraprocessed = []
        detected_subjects_ultraprocessed = []
        grades_ultraprocessed = []
        confidences_ultraprocessed = []

        for index, (prediction, box) in enumerate(zip(true_predictions, true_boxes)):
            predicted_label = iob_to_label(prediction).lower()
            detection_line = get_line_id(line_2_bbox, box)

            label_flag = False
            if (
                predicted_label in answers_list
                and detection_line not in lines_with_detected_subjects
            ):
                # Legit pre-process
                preprocessed = re.sub(",", ".", words[0][index])
                # XXX Ultra-processed?
                preprocessed = re.sub("[^0-9]", "", preprocessed)

                # XXX Still needed?
                if "," in preprocessed:
                    preprocessed = preprocessed.replace(",", ".")

                # XXX This also ultra-process the model output (only floats allowed)
                try:
                    label_flag = True
                    grade = float(preprocessed)
                    grades_tags.append(predicted_label)
                    grades.append(grade)
                    confidences.append(conf[0][index])
                    detected_subjects.append(predicted_label[:-7])
                    lines_with_detected_subjects.append(detection_line)

                    # Ultra process the grades, ie:
                    if grade >= 0 and grade <= 10:
                        detected_subjects_ultraprocessed.append(predicted_label[:-7])
                        grades_ultraprocessed.append(grade)
                        confidences_ultraprocessed.append(conf[0][index])

                except ValueError:
                    label_flag = False
                    pass

            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            text_xy = (int(box[0]) + 10, int(box[1]) + 10)

            color = label2color[predicted_label]
            img = cv2.rectangle(img, start_point, end_point, color, 2)
            
            if label_flag:
                img = cv2.putText(
                    img, predicted_label, text_xy, font, font_scale, color, 1, cv2.LINE_AA
                )

        save_results_path = os.path.join(os.getcwd(), "results")

        if not os.path.isdir(save_results_path):
            os.mkdir(save_results_path)

        img_name = "".join(["result_", str(iter), ".png"])
        img_name = os.path.join(save_results_path, img_name)
        cv2.imwrite(img_name, img)

        grades_output = postprocess_outputs(
            detected_subjects, grades_tags, grades, confidences, iter
        )
        grades_output_ultraprocessed = postprocess_outputs(
            detected_subjects_ultraprocessed,
            grades_tags,
            grades_ultraprocessed,
            confidences_ultraprocessed,
            iter,
            ultra=True,
        )

        for grade in grades_output:
            print(grade, grades_output[grade]["grade"])
        print("")
        for grade in grades_output_ultraprocessed:
            print(grade, grades_output_ultraprocessed[grade]["grade"])

        if OBTAIN_METRICS:
            num_errors, num_gt_grades = check_ground_truth(
                grades_dict=grades_output, json_path=img_path
            )
            total_errors += num_errors
            total_gt_grades += num_gt_grades

            num_errors_ultraprocessed, _ = check_ground_truth(
                grades_dict=grades_output_ultraprocessed, json_path=img_path
            )
            total_errors_ultraprocessed += num_errors_ultraprocessed

            ultra_errors_string = "".join(
                [
                    "File num.",
                    str(iter).zfill(4),
                    " ultra-processed errors: ",
                    "".join(
                        [str(num_errors_ultraprocessed), "/", str(num_gt_grades)]
                    ).rjust(9),
                ]
            )
            errors_string = "".join(
                [
                    "File num.",
                    str(iter).zfill(4),
                    " errors: ",
                    "".join([str(num_errors), "/", str(num_gt_grades)]).rjust(25),
                ]
            )

            print(errors_string)
            print(ultra_errors_string, "\n")

            documents_stats.append(num_errors)

        if SEND_TO_DATABASE:
            # Create Required Data
            # TODO: Obtain (where possible) from Database

            student = spawn_student()
            academic_year = spawn_year()
            id_curriculo = spawn_id_curriculo()
            base_id = random.randint(20, 1000)

            for i, subject in enumerate(grades_output):
                id = base_id + i
                create_subject_in_database(
                    subject_name=subject,
                    id_subject=id,
                    id_academic_year=academic_year["id_curso"],
                    id_curriculo=id_curriculo,
                )

                post_subject_grade(
                    num_grade=grades_output[subject],
                    subject_id=id,
                    student_id=student["clave_comillas"],
                    curriculo_id=id_curriculo,
                )

        iter += 1


postproc_docs_stats = postprocess_docs_stats(stats=documents_stats)
generate_plot(stats=postproc_docs_stats, path=save_results_path)
generate_aggregated_metrics(
    errors=total_errors, total=total_gt_grades, path=save_results_path
)
generate_aggregated_metrics(
    errors=total_errors_ultraprocessed,
    total=total_gt_grades,
    path=save_results_path,
    ultra=True,
)

if SEND_TO_DATABASE:
    out_getter = requests.get("http://localhost:8000/Students/202303542")
    print(out_getter.json())
