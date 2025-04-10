import os
import json
import random
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class AudioInstruction(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, audio_dir, vis_root, ann_path, prompt_test=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        audio_dir (string): Root direction of audio
        """
        self.vis_root = vis_root
        self.audio_dir = audio_dir
        self.fallback_audio_dir = '../Data/medtrinity/audio_wav'

        self.vis_processor = vis_processor

        self.text_processor = text_processor
        self.audio_processor = audio_processor

        # path: ../data/LISA/json/train.json
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
            
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['img_name']
        image_id, _ = os.path.splitext(info['img_name'])
        image_path = os.path.join(self.vis_root, image_file)

        if not os.path.exists(image_path):
            image_id = image_file.split('_')[0]
            image_path = os.path.join(self.vis_root, f"{image_id}.jpg")


        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)


        # If we have multiple outputs/queries, randomly pick one
        if isinstance(info['answer'], list) and len(info['answer']) > 1:
            number = random.randint(0, len(info['answer']) - 1)  # Select a random index for query/output

            answer = info['answer'][number]  # Select the corresponding answer
            text_question = info['question'][number]  # Select the corresponding query

            # Finding correct image and audio paths accordingly
            image_path = os.path.join(self.vis_root, f'{image_id}.jpg') 
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)

            # audio
            audio_path = os.path.join(self.audio_dir, f'{image_id}_q{number + 1}.wav')  # Assuming audio file naming starts from 1
            if not os.path.exists(audio_path):
                audio_path = self.get_random_audio_path()
                
            waveform, sample_rate = torchaudio.load(audio_path)

        else:
            answer = info['answer']  # Single answer
            text_question = info['question']  # Single query

            # audio
            audio_file, _ = os.path.splitext(info['img_name'])
            audio_path = os.path.join(self.audio_dir, f'{audio_file}.wav')
            if not os.path.exists(audio_path):
                audio_path = self.get_random_audio_path()

            waveform, sample_rate = torchaudio.load(audio_path) 

            self.COT = "You are a medical assistant helping us analyze the provided images and answer queries from audio. For each query, follow these steps:\
            1. Image Analysis: Examine the image carefully, identifying key objects, patterns, or medical/technical details.\
            2. Audio Processing: Understand the query from speech by breaking it down into sub-components, identifying its intent.\
            3. Step-by-Step Reasoning: Use logical steps to infer the correct response based on available data.\
            4. Final Answer: Provide a clear and well-structured response based on the analysis. The response must include both the answer and an explanation."

            self.TOT = "You are a medical assistant helping us analyze the provided images and answer queries from audio. \
            To ensure optimal accuracy, consider multiple reasoning paths before choosing the best response. Follow this structured process:\
            1. Image & Audio Processing\
                - Extract features from the image and detect key patterns.\
                - Understand the query from audio.\
            2. Branching Reasoning Paths\
                - Path 1: Direct Answering → If the answer is straightforward, generate a response.\
                - Path 2: Context-Based Inference → If additional reasoning is required, infer based on related data.\
                - Path 3: Uncertainty Handling → If confidence is low, outline possible interpretations.\
            3. Evaluation of Paths\
                - Rank the paths based on correctness, confidence, and clarity.\
                - Select the best path and refine the response.\
            4. Final Answer with Justification\
                - Provide a clear and well-structured response based on the analysis. The response must include both the answer and an explanation."


        # # For audio instruction
        # instruction = "<Img><ImageHere></Img> You are a medical assistant helping us analyze medical images." 
        instruction = "<Img><ImageHere></Img> You are a medical assistant helping us analyze the provided images and answer queries from audio." 
        # instruction = f"<Img><ImageHere></Img> {self.TOT}"
        # # For text instruction
        # instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        waveform_array = waveform.squeeze().numpy()

        waveform = self.audio_processor(waveform_array) #, sampling_rate=16000, return_tensors="pt").input_features
        waveform = waveform.squeeze()

        return {
            "image": image,
            "audio": waveform, #torch.rand(80, 3000, dtype=torch.float16), # double length of the max_source_positions
            "instruction_input": instruction,
            "answer": answer,
            "image_id": image_id,
            "question": text_question,
        }

    def get_random_audio_path(self):
        """
        Get a random audio file from the fallback audio directory.
        """
        if not self.fallback_audio_dir:
            raise ValueError("Fallback audio directory is not provided.")
        
        # List all .wav files in the fallback directory
        audio_files = [f for f in os.listdir(self.fallback_audio_dir) if f.endswith('.wav')]
        if not audio_files:
            raise ValueError("No audio files found in the fallback directory.")
        
        # Select a random audio file
        random_audio_file = random.choice(audio_files)
        return os.path.join(self.fallback_audio_dir, random_audio_file)




# COT = "You are a medical assistant helping us analyze the provided images and answer queries from audio. For each query, follow these steps:\
# 1. Image Analysis: Examine the image carefully, identifying key objects, patterns, or medical/technical details.\
# 2. Audio Processing: Understand the query from speech by breaking it down into sub-components, identifying its intent.\
# 3. Step-by-Step Reasoning: Use logical steps to infer the correct response based on available data.\
# 4. Final Answer: Provide a clear and well-structured response based on the analysis. The response must include both the answer and an explanation."

# TOT = "You are a medical assistant helping us analyze the provided images and answer queries from audio. \
# To ensure optimal accuracy, consider multiple reasoning paths before choosing the best response. Follow this structured process:\
# 1. Image & Audio Processing\
#     - Extract features from the image and detect key patterns.\
#     - Understand the query from audio.\
# 2. Branching Reasoning Paths\
#     - Path 1: Direct Answering → If the answer is straightforward, generate a response.\
#     - Path 2: Context-Based Inference → If additional reasoning is required, infer based on related data.\
#     - Path 3: Uncertainty Handling → If confidence is low, outline possible interpretations.\
# 3. Evaluation of Paths\
#     - Rank the paths based on correctness, confidence, and clarity.\
#     - Select the best path and refine the response.\
# 4. Final Answer with Justification\
#     - Provide a clear and well-structured response based on the analysis. The response must include both the answer and an explanation."















# import os
# import json
# import random
# import torch
# import torchaudio
# from PIL import Image
# from torch.utils.data import Dataset

# class AudioInstruction(Dataset):
#     def __init__(self, vis_processor, text_processor, audio_processor, audio_dir, vis_root, ann_path, prompt_test=None):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): Directory to store the annotation file.
#         audio_dir (string): Root directory of audio files.
#         """
#         self.vis_root = vis_root
#         self.audio_dir = audio_dir
#         self.fallback_audio_dir = '../Data/medtrinity/audio_wav'
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor
#         self.audio_processor = audio_processor

#         # Supported image extensions
#         self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

#         with open(ann_path, 'r') as f:
#             self.ann = json.load(f)
            
#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):
#         info = self.ann[index]

#         image_file = info['img_name']
#         image_id, _ = os.path.splitext(image_file)
        
#         # Try finding the image with various extensions
#         image_path = self.find_image_with_extensions(image_id)

#         if image_path is None:
#             raise FileNotFoundError(f"Image {image_id} not found in {self.vis_root} with supported extensions.")

#         image = Image.open(image_path).convert("RGB")
#         image = self.vis_processor(image)

#         # Handling multiple outputs/queries
#         if isinstance(info['answer'], list) and len(info['answer']) > 1:
#             number = random.randint(0, len(info['answer']) - 1)  
#             answer = info['answer'][number]
#             text_question = info['question'][number]

#             # Find the corresponding image again
#             image_path = self.find_image_with_extensions(image_id)
#             image = Image.open(image_path).convert("RGB")
#             image = self.vis_processor(image)

#             # Audio file handling
#             audio_path = os.path.join(self.audio_dir, f'{image_id}_q{number + 1}.wav')
#             if not os.path.exists(audio_path):
#                 audio_path = self.get_random_audio_path()
                
#             waveform, sample_rate = torchaudio.load(audio_path)

#         else:
#             answer = info['answer']
#             text_question = info['question']

#             # Audio file handling
#             audio_path = os.path.join(self.audio_dir, f'{image_id}.wav')
#             if not os.path.exists(audio_path):
#                 audio_path = self.get_random_audio_path()

#             waveform, sample_rate = torchaudio.load(audio_path) 
        
#         instruction = "<Img><ImageHere></Img> You are a general assistant helping us analyze the provided images and answer queries from audio." 

#         waveform_array = waveform.squeeze().numpy()
#         waveform = self.audio_processor(waveform_array)
#         waveform = waveform.squeeze()

#         return {
#             "image": image,
#             "audio": waveform,
#             "instruction_input": instruction,
#             "answer": answer,
#             "image_id": image_id,
#             "question": text_question,
#         }

#     def find_image_with_extensions(self, image_id):
#         """
#         Tries to find an image file with different supported extensions.
#         """
#         for ext in self.image_extensions:
#             image_path = os.path.join(self.vis_root, f"{image_id}{ext}")
#             if os.path.exists(image_path):
#                 return image_path
#         return None

#     def get_random_audio_path(self):
#         """
#         Get a random audio file from the fallback audio directory.
#         """
#         if not self.fallback_audio_dir:
#             raise ValueError("Fallback audio directory is not provided.")
        
#         audio_files = [f for f in os.listdir(self.fallback_audio_dir) if f.endswith('.wav')]
#         if not audio_files:
#             raise ValueError("No audio files found in the fallback directory.")
        
#         return os.path.join(self.fallback_audio_dir, random.choice(audio_files))
