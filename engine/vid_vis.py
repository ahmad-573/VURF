import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
# from IPython.core.display import HTML

import json
from tqdm import tqdm

from torchvision import transforms
from lavis.processors import transforms_video
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA

from engine.utils import ProgramGenerator, ProgramInterpreter
from prompts.star import create_prompt

import torch

import pickle

def sevila_demo(video="", 
    question="", interval=None, 
    option1="?", option2="?", option3="?", option4="?",
    video_frame_num=32, 
    keyframe_num=4):
    
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'
    

    global sevila 
    if sevila:
        if device == "cpu":
            sevila = sevila.float()
        else:
            sevila = sevila.to(int(device))
        
    vpath = video 
    image_size = 224
    raw_clip, indice, fps, vlen = load_video_demo(
        video_path=vpath,
        n_frms=int(video_frame_num),
        height=image_size,
        width=image_size,
        sampling="uniform",
        clip_proposal=interval
    )
    video_len = vlen/fps
    return raw_clip, video_len

    clip = transform(raw_clip.permute(1,0,2,3))
    clip = clip.float().to(int(device))
    clip = clip.unsqueeze(0)

    if option1[-1] != '.':
        option1 += '.'
    if option2[-1] != '.':
        option2 += '.' 
    if option3[-1] != '.':
        option3 += '.'
    if option4[-1] != '.':
        option4 += '.'
    option_dict = {0:option1, 1:option2, 2:option3, 3:option4}
    options = 'Option A:{} Option B:{} Option C:{} Option D:{}'.format(option1, option2, option3, option4)
    text_input_qa = 'Question: ' + question + ' ' + options + ' ' + QA_prompt
    text_input_loc = 'Question: ' + question + ' ' + options + ' ' + LOC_propmpt

    # return raw_clip, clip, video_len
    
    out = sevila.generate_demo(clip, text_input_qa, text_input_loc, int(keyframe_num))
    # print(out)
    answer_id = out['output_text'][0]
    answer = option_dict[answer_id]
    select_index = out['frame_idx'][0]
    # images = [] 
    keyframes = []
    timestamps =[]
    
    # print('raw_clip', len(raw_clip))
    # for j in range(int(video_frame_num)):
    #     image = raw_clip[:, j, :, :].int()
    #     image = image.permute(1, 2, 0).numpy() 
    #     images.append(image)
    
    video_len = vlen/fps # seconds
    
    for i in select_index:
        image = raw_clip[:, i, :, :].int()
        image = image.permute(1, 2, 0).numpy() 
        keyframes.append(image)
        select_i = indice[i]
        time = round((select_i / vlen) * video_len, 2)
        timestamps.append(str(time)+'s')
    
    gr.components.Gallery(keyframes)
    #gr.components.Gallery(images)
    timestamps_des = ''
    for i in range(len(select_index)):
        timestamps_des += 'Keyframe {}: {} \n'.format(str(i+1), timestamps[i])
    
    return keyframes, timestamps_des, answer

if __name__ == '__main__':
    # print("tadaaa")
    img_size = 224
    num_query_token = 32
    t5_model = 'google/flan-t5-xl'
    drop_path_rate = 0
    use_grad_checkpoint = False
    vit_precision = "fp16"
    freeze_vit = True
    prompt = ''
    max_txt_len = 77
    answer_num = 5
    apply_lemmatizer = False
    task = 'freeze_loc_freeze_qa_vid'

    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'

    # prompt
    LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
    QA_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'

    # processors config
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    image_size = img_size
    transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

    interpreter = ProgramInterpreter()

    print('Model Loading \nLoading the SeViLA model can take a few minutes (typically 2-3).')
    # sevila = SeViLA(
    #     img_size=img_size,
    #     drop_path_rate=drop_path_rate,
    #     use_grad_checkpoint=use_grad_checkpoint,
    #     vit_precision=vit_precision,
    #     freeze_vit=freeze_vit,
    #     num_query_token=num_query_token,
    #     t5_model=t5_model,
    #     prompt=prompt,
    #     max_txt_len=max_txt_len,
    #     apply_lemmatizer=apply_lemmatizer,
    #     frame_num=4,
    #     answer_num=answer_num,
    #     task=task,
    #         )

    # sevila.load_checkpoint(url_or_filename='https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth')
    # print('Model Loaded')
    sevila = None


    ANS_MAPPING = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E'}

    # val_path = "/share/data/drive_5/ahmad/val_shuffled.json"
    val_path = "/share/data/drive_5/ahmad/val_shuffled_2.json"
    videos = json.load(open(val_path))
    
    correct = 0
    total = 0
    num_errors = 0

    with open('progs_vgpt.pkl', 'rb') as file:
        loaded_list = pickle.load(file)

    counter = 0
    for vid in tqdm(videos[1000:2000]):
        vid_path = vid["vid_path"]
        question = vid["question"]
        options = vid["options"]
        answer_id = vid["answer"]

        vid_path = '/share/users/ahmad/visprog/BGR.mp4'

        prog_h = loaded_list[counter]
        counter += 1

        # assert(vp == vid_path)

        transcript = ""
        if "transcript" in vid.keys():
            transcript = vid["transcript"]
        
        interval = None
        # if "start" in vid.keys():
        #     start = vid["start"]
        #     end = vid["end"]
        #     interval = (start, end)

        raw_clip, vlen = sevila_demo(video=vid_path, question=question, option1=options[0], option2=options[1], option3=options[2], option4=options[3], interval=interval,video_frame_num=128, keyframe_num=4)

        # print("VLEN", vlen, "FPS", fps)
        # full_time = vlen / fps

        print("video:", vid_path, " of time:", vlen)
        print("Question:", question)
        
        frames = []
        for frame in raw_clip.permute(1,0,2,3):
            tr = transforms.ToPILImage()
            frame_pil = tr(frame / 255.0)
            frames.append(frame_pil)


        os.environ["OPENAI_API_KEY"] = "INSERT_KEY_HERE"
        init_state = dict(
            VIDEO=frames,
            SUM_PATH=vid_path,
            OPTIONS=options,
            SUM_INT=interval,
            FULL_TIME=vlen,
            TRANSCRIPT=transcript
        )
        # try:
        # generator = ProgramGenerator(prompter=create_prompt)
        # prog,_ = generator.generate(dict(question=question))
        prog_h = 'T0=TRACK(video=VIDEO,query="hand",max_tracks=100000)'
        print(prog_h)
        result, prog_state = interpreter.execute(prog_h,init_state,inspect=False)
        
        print(result)
        break

        if isinstance(result, tuple) and result[0] == 'vqa':
            vid_var = result[1]
            vid = prog_state[vid_var]
            r_clip = []
            for iterr, fr in enumerate(vid):
                tr = transforms.PILToTensor()
                r_clip.append(tr(fr))
                # fr.save(f"tests/frame_{iterr}.png")
            
            if len(r_clip) == 0:
                print("COPYING RAW CLIP FOR VQA SINCE INPUT WAS EMPTY")
                clip = transform(raw_clip.permute(1,0,2,3))
            else:
                clip = transform(torch.stack(r_clip))

            clip = clip.float().to(int(device))
            clip = clip.unsqueeze(0)

            print("Going VQA with shape: ", clip.shape)

            option1=options[0] 
            option2=options[1]
            option3=options[2]
            option4=options[3]

            if (not option1) or (option1[-1] != '.'):
                option1 += '.'
            if (not option2) or option2[-1] != '.':
                option2 += '.' 
            if (not option3) or option3[-1] != '.':
                option3 += '.'
            if (not option4) or option4[-1] != '.':
                option4 += '.'
            option_dict = {0:option1, 1:option2, 2:option3, 3:option4}
            options = 'Option A:{} Option B:{} Option C:{} Option D:{}'.format(option1, option2, option3, option4)
            text_input_qa = 'Question: ' + question + ' ' + options + ' ' + QA_prompt
            text_input_loc = 'Question: ' + question + ' ' + options + ' ' + LOC_propmpt
            
            out = sevila.generate_demo(clip, text_input_qa, text_input_loc, 4)
            # print(out)
            pred = out['output_text'][0]
            pred_answer = option_dict[pred]

            print("Actual Answer:", answer_id, option_dict[answer_id])
            print("Predicted Answer:", pred, pred_answer)

            total += 1
            if answer_id == pred:
                correct += 1

            # if total % 10 == 0:
            print("Current Accuracy:", correct / total, " after total vids: ", total, " with errors:", num_errors)

        else:
            option1=options[0] 
            option2=options[1]
            option3=options[2]
            option4=options[3]
            option_dict = {0:option1, 1:option2, 2:option3, 3:option4}
            try:
                pred = int(result)
            except:
                pred = 0
                idxx = ['0','1', '2', '3']
                for ch in result:
                    if ch in idxx:
                        pred = int(ch) 
                        break
            pred_answer = options[pred]
            print("Actual Answer:", answer_id, option_dict[answer_id])
            print("Predicted Answer:", pred, pred_answer)

            total += 1
            if answer_id == pred:
                correct += 1

            # if total % 10 == 0:
            print("Current Accuracy:", correct / total, " after total vids: ", total, " with errors:", num_errors)
        
        # except Exception as error:
        #     print("Error from interpreter:", error)
        #     num_errors += 1
        #     total += 1
        #     print("Current Accuracy:", correct / total, " after total vids: ", total)
        
        break
        if num_errors >= 50:
            print("ENDING!! ERRORS TOO MUCH")
            break

