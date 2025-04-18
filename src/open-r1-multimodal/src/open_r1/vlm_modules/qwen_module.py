from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.utils.logger_utils import get_logger
import os
import re
log_path = os.getenv("LOG_PATH")

logger = get_logger(name="qwen module log", log_file=log_path.replace(".txt", ".log"))
POTENTIAL_LIST = ["cad", "chf", "mi"]
def process_predict_label(content):
    # try:
    #     parsed = ast.literal_eval(content)
    #     if isinstance(parsed, list):
    #         return [item.strip().lower() for item in parsed]
    # except:
    #     pass

    # if "|" in content:
    #     return [item.strip().lower() for item in content.split("|")]

    # return [content.strip().lower()]
    result = []
    ct = content.lower()
    for d in POTENTIAL_LIST:
        if d in ct:
            result.append(d)

    return result

    


def process_true_labels(raw):
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]  

    if isinstance(raw, str):
        return [item.strip() for item in raw.split('|') if item.strip()]

    return []

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "cvd":
                return """{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>. The answer is in ["CAD", "CHF", "MI"], use "|" to separate multiple answers."""
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def format_reward(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        # logger.info(f"format reward: contents: {completion_contents}")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def exact_match_reward(completions, **kwargs):
        import re

        true_labels_batch = kwargs.get("solution", [])
        completion_contents = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, true_labels in zip(completion_contents, true_labels_batch):
            # Extract the predicted label string from <answer>...</answer>
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
            pred_labels = []
            logger.info(f"original predict result: {content}")
            if match:
                pred_labels = process_predict_label(match.group(1).strip())
            
            true_labels = true_labels.replace("<answer>", "").replace("</answer>", "").strip()
            true_labels = [label.strip().lower() for label in true_labels.split("|")]
       
            # Compare the predicted and true label sets
            is_match = set(pred_labels) == set(true_labels)
            
            rwd = 1.0 if is_match else 0.0
            if len(pred_labels) == 0:
                logger.info(f"no predict label: {content}")
            else:
                logger.info(f"exact_match_reward true_label: {true_labels} \n  pred_labels: {pred_labels} \n reward: {rwd}")

            logger.info(f"*content*: {content} \n  *true_label*: {true_labels} \n *pred_labels*: {pred_labels}")
            # if len(pred_labels) == 0:
            #     logger.info(f"bad format: {content} \n")
            rewards.append(rwd)

        return rewards

    @staticmethod
    def f1_score_reward(completions, **kwargs):

        import re
        from sklearn.metrics import f1_score

        true_labels_batch = kwargs.get("solution", [])  # Ground truth labels
        completion_contents = [completion[0]["content"] for completion in completions]

        rewards = []
        for content, true_labels in zip(completion_contents, true_labels_batch):
            # Extract predicted labels
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)

            pred_labels = []
            if match:
                pred_labels = process_predict_label(match.group(1).strip())

            true_labels = true_labels.replace("<answer>", "").replace("</answer>", "").strip()
            true_labels = [label.strip().lower() for label in true_labels.split("|")]
            

            # Union label set from both true and predicted labels
            label_set = sorted(set(true_labels + pred_labels))

            # Binary indicator vectors
            y_true = [1 if l in true_labels else 0 for l in label_set]
            y_pred = [1 if l in pred_labels else 0 for l in label_set]

            # F1 calculation
            rwd = 0
            if sum(y_true) == 0 and sum(y_pred) == 0:
                rwd = 1.0
            else:
                rwd = f1_score(y_true, y_pred)
            logger.info(f"f1_match_reward true_label: {true_labels} \n  pred_labels: {pred_labels} \n reward: {rwd}")
            if len(pred_labels) == 0:
                logger.info(f"bad format: {content} \n")
            rewards.append(rwd)
        return rewards
    
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "cvd":
                    return Qwen2VLModule.exact_match_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "cvd":
                    return Qwen2VLModule.format_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "f1":
            match task_type:
                case "cvd":
                    return Qwen2VLModule.f1_score_reward    
        else:
            raise ValueError(f"Unsupported reward function: {func}")
