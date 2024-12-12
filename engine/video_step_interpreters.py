import cv2
import os
import torch
import openai
from openai import OpenAI
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
# from transformers import (ViltProcessor, ViltForQuestionAnswering, 
#     OwlViTProcessor, OwlViTForObjectDetection,
#     MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
#     CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
# from diffusers import StableDiffusionInpaintPipeline

import datetime
import time
from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame

from argparse import ArgumentParser

from tqdm import tqdm

#videochatgpt
from Video_ChatGPT import summarize
from Video_ChatGPT.video_chatgpt.eval.model_utils import initialize_model, load_video

def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result

class TrackInterpreter():
    step_name = "TRACK"
    def __init__(self):
        print(f'Registering {self.step_name} step')
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        np.random.seed(42)
        """
        Arguments loading
        """
        parser = ArgumentParser()

        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser)
        self.deva_model, self.cfg, self.args = get_model_and_config(parser)
        self.gd_model, self.sam_model = get_grounding_dino_model(self.cfg, 'cuda')

        self.cfg['temporal_setting'] = self.args.temporal_setting.lower()
        assert self.cfg['temporal_setting'] in ['semionline', 'online']
        

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        vid_var = args['video']
        query = eval(args['query'])
        max_tracks = eval(args['max_tracks'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var,query,max_tracks,output_var

    def execute(self, prog_step, inspect=False):
        vid_var,query,max_tracks,output_var = self.parse(prog_step)
        vid = prog_step.state[vid_var]

        # update to infinity
        if max_tracks == -1:
            max_tracks = 1000000

        out_path = self.cfg['output']
        self.cfg['prompt'] = query
        self.cfg['max_num_objects'] = max_tracks

        # Start eval
        vid_length = len(vid)
        # print("Frames going in track: ", vid_length)
        # no need to count usage for LT if the video is not that long anyway
        self.cfg['enable_long_term_count_usage'] = (
            self.cfg['enable_long_term']
            and (vid_length / (self.cfg['max_mid_term_frames'] - self.cfg['min_mid_term_frames']) *
                self.cfg['num_prototypes']) >= self.cfg['max_long_term_elements'])

        print('Configuration:', self.cfg)

        deva = DEVAInferenceCore(self.deva_model, config=self.cfg)
        deva.next_voting_frame = self.cfg['num_voting_frames'] - 1
        deva.enabled_long_id()
        result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            for ti, frame in enumerate(tqdm(vid)):
                frame_np = np.array(frame)
                process_frame(deva, self.gd_model, self.sam_model, f'frame_{ti}', result_saver, ti, image_np=frame_np)
            flush_buffer(deva, result_saver)
        result_saver.end()

        result = result_saver.video_json['annotations']

        boxes_by_id = {}
        for num, frame_res in enumerate(result):
            bboxes = frame_res['bboxes']
            segs = frame_res['segments_info']
            if len(bboxes) == 0:
                continue
            assert len(bboxes) == len(segs)
            for i in range(len(bboxes)):
                b_id = segs[i]['id']
                box = bboxes[i]
                if b_id in boxes_by_id:
                    boxes_by_id[b_id].append((num, box))
                else:
                    boxes_by_id[b_id] = [(num, box)]

        final_boxes = []
        for b_id in boxes_by_id:
            final_boxes.append(boxes_by_id[b_id])

        if len(final_boxes) > 0:
            print("\nNumber of frames in first track: ", len(final_boxes[0]))
        else:
            print("No object detected!")
            # final_boxes = [[(num, [0,0,100,100])]]

        prog_step.state[output_var] = final_boxes
        return final_boxes
        # print(len(result['annotations'])) 

class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]
    
    def expand_box_precise(self, box, max_x, max_y, add_x, add_y):
        x1,y1,x2,y2 = box
        
        if x2 + int(add_x / 2) > max_x:
            x2_new = max_x
            x1_new = x1 - (add_x - (max_x - x2))
        elif x1 - int(add_x / 2) < 0:
            x1_new = 0
            x2_new = x2 + (add_x - x1)
        else:
            x2_new = x2 + int(add_x / 2)
            x1_new = x1 - (add_x - (x2_new - x2))

        if y2 + int(add_y / 2) > max_y:
            y2_new = max_y
            y1_new = y1 - (add_y - (max_y - y2))
        elif y1 - int(add_y / 2) < 0:
            y1_new = 0
            y2_new = y2 + (add_y - y1)
        else:
            y2_new = y2 + int(add_y / 2)
            y1_new = y1 - (add_y - (y2_new - y2))

        return [x1_new,y1_new, x2_new, y2_new]

        

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        track_var = parse_result['args']['track']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var,track_var,output_var
    
    def execute(self, prog_step, inspect=False):
        vid_var,track_var,output_var = self.parse(prog_step)
        vid = prog_step.state[vid_var]
        track = prog_step.state[track_var]
        if len(track) > 0:
            track = track[0]
        else:
            print("CROP got no track so returning full video")
            prog_step.state[output_var] = vid
            return vid

        final_frames = []
        w, h = 0, 0
        for num, box in track:
            x1,y1,x2,y2 = box
            w = max(w, x2 - x1)
            h = max(h, y2 - y1)
        
        print("CROPPING TO:", w,h)
        for num, box in track:
            x1,y1,x2,y2 = box
            max_x, max_y = np.array(vid[num]).shape[0] - 1, np.array(vid[num]).shape[1] - 1
            add_x = w - (x2 - x1)
            add_y = h - (y2 - y1)
            box_exp = self.expand_box_precise(box, max_x, max_y, add_x, add_y)
            frame = vid[num].crop(box_exp)
            final_frames.append(frame.resize((224,224)))

        prog_step.state[output_var] = final_frames
        return final_frames     


class TrimInterpreter():
    step_name = "TRIM"

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        # interval = eval(parse_result['args']['interval'])
        start = parse_result['args']['start']
        end = parse_result['args']['end']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var,start,end,output_var
    
    def execute(self, prog_step, inspect=False):
        vid_var, start, end, output_var = self.parse(prog_step)
        vid = prog_step.state[vid_var]
        num_frames = len(vid)

        # start = num_frames * interval[0]
        # end = int(num_frames * interval[1]) - 1
        # print(prog_step.state)
        try:
            start_time = prog_step.state[start]
        except:
            start_time = eval(start)
        try:
            end_time = prog_step.state[end]
        except:
            end_time = eval(end)
        
        start = int(num_frames * start_time)
        end = int(num_frames * end_time) - 1

        if (end - start + 1) < 5:
            end = min(num_frames - 1, end + 5)
            start = max(0,start - 5)
        trimmed = vid[start:(end + 1)]
        prog_step.state[output_var] = trimmed
        return trimmed

class TrimAfterInterpreter():
    step_name = "TRIM_AFTER"

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        interval = eval(parse_result['args']['interval'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var,interval,output_var
    
    def execute(self, prog_step, inspect=False):
        vid_var, interval, output_var = self.parse(prog_step)
        vid = prog_step.state[vid_var]
        num_frames = len(vid)

        start = num_frames * interval[0]
        end = int(num_frames * interval[1]) - 1

        center = (start + end) // 2
        shift = min(center - start, num_frames - 1 - end)

        start += shift
        end += shift
        end = min(num_frames - 1, end + 2 * shift)

        trimmed = vid[start:(end + 1)]
        prog_step.state[output_var] = trimmed
        return trimmed

class TrimBeforeInterpreter():
    step_name = "TRIM_BEFORE"

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        interval = eval(parse_result['args']['interval'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var,interval,output_var
    
    def execute(self, prog_step, inspect=False):
        vid_var, interval, output_var = self.parse(prog_step)
        vid = prog_step.state[vid_var]
        num_frames = len(vid)

        start = num_frames * interval[0]
        end = int(num_frames * interval[1]) - 1

        center = (start + end) // 2
        shift = min(end - center, start)

        start += shift
        end += shift
        start = max(0, start - 2 * shift)

        trimmed = vid[start:(end + 1)]
        prog_step.state[output_var] = trimmed
        return trimmed 


class MergeInterpreter():
    step_name = "MERGE"

    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        track1_var = parse_result['args']['track1']
        track2_var = parse_result['args']['track2']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return track1_var, track2_var, output_var

    def merge(self, b1, b2):
        x1 = min(b1[0], b2[0])
        x2 = max(b1[2], b2[2])
        y1 = min(b1[1], b2[1])
        y2 = max(b1[3], b2[3])

        return (x1,y1,x2,y2)

    def execute(self, prog_step, inspect=False):
        track1_var, track2_var, output_var = self.parse(prog_step)
        track1_tmp = prog_step.state[track1_var]
        track2_tmp = prog_step.state[track2_var]

        # print(len(track1_tmp))
        # print(len(track2_tmp))

        track1 = []
        track2 = []
        if len(track1_tmp) > 0:
            track1 = track1_tmp[0]
        if len(track2_tmp) > 0:
            track2 = track2_tmp[0]
        

        final_track = []
        if len(track1) == 0:
            final_track = track2
        elif len(track2) == 0:
            final_track = track1
        else:
            pairs = {}
            for num, box in track1:
                pairs[num] = [box]

            for num, box in track2:
                if num in pairs:
                    pairs[num].append(box)
                else:
                    pairs[num] = [box]

            for frame in pairs:
                pair = pairs[frame]
                if len(pair) == 1:
                    final_track.append((frame, pair[0]))
                else:
                    final_track.append((frame, self.merge(pair[0], pair[1])))

        prog_step.state[output_var] = [final_track]
        return [final_track]


class CountInterpreter():
    step_name = "COUNT"

    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        track_var = parse_result['args']['track']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return track_var, output_var
    
    def execute(self, prog_step, inspect=False):
        track_var, output_var = self.parse(prog_step)
        track = prog_step.state[track_var]

        prog_step.state[output_var] = len(track)
        return len(track)



class SummaryInterpreter():
    step_name = "SUMMARIZE"

    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        # query = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var, output_var
    
    def execute(self, prog_step, inspect=False):
        print("SUMMARY")
        vid_var, output_var = self.parse(prog_step)
        
        args = summarize.parse_args()
        
        model, vision_tower, tokenizer, image_processor, video_token_len = \
            initialize_model(args.model_name, args.projection_path)

        # video_path = args.video_path
        video_path = prog_step.state["SUM_PATH"]
        interval = prog_step.state["SUM_INT"]

        print("INTERVAL", interval)
        if os.path.exists(video_path):
            video_frames = load_video(video_path, clip_proposal=interval)
        
        print("VIDEO SHAPE", len(video_frames), type(video_frames[0]))
        save_path = "sum_test/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for it, vf in enumerate(video_frames):
            vf.save(save_path + f"frame_{it}.jpg")
        conv_mode = args.conv_mode

        # try:
        question = f"Describe in detail what is happening in this video."
        print("GOING INTO VGPT:", question)
        # Run inference on the video and add the output to the list
        output = summarize.video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len)

        # prompt = f"Given a video summary describing the following context of the video, please generate a narrative or description of what might happen next in the video. \
        # Predict the events, actions, or outcomes that could occur in the video. Limit your answer to 50 words.\
        # \nVideo Summary: {output}"

        # print("\n\n", output)
        print(output)
        prog_step.state[output_var] = output
        return output

        """
        add the openai code here
            """

        # except Exception as e:
        #     print(f"Error processing video file '{video_path}': {e}")

class PredictInterpreter():
    step_name = "PREDICT"

    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        summary_var = parse_result['args']['summary']
        query = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return summary_var, query,  output_var
    
    def execute(self, prog_step, inspect=False):
        summary_var, query,  output_var = self.parse(prog_step)
        summary = prog_step.state[summary_var]

        os.environ["OPENAI_API_KEY"] = "INSERT_KEY_HERE"

        options = prog_step.state['OPTIONS']
        
        opt = f'OPTION A: {options[0]}, OPTION B: {options[1]}, OPTION C: {options[2]}, OPTION D: {options[3]}'

        prompt = f"Given a video summary describing the following context of the video, please answer the following question about what might happen next in the video.  \
            Only choose from the given options. Just output the option number (i.e. 0 for option A, 1 for option B, 2 for option C and 3 for option D.)\
            \nVideo Summary: {summary} \
            \nQuestion: {query}\
            \nOptions: {opt}"

        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=prompt,
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1,
        #     logprobs=1
        # )
        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        )

        completion = client.completions.create( # Change the method
            model = "gpt-3.5-turbo-instruct",
            prompt = prompt,
            temperature=0.7,
            max_tokens=512,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1
        )


        # prog = completion.choices[0].text.lstrip('\n').rstrip('\n')
        # return prog, 0
        
        answer = completion.choices[0].text.lstrip('\n').rstrip('\n')
        prog_step.state[output_var] = answer
        return answer

class AnalyseInterpreter():
    step_name = "ANALYSE"

    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        summary_var = parse_result['args']['summary']
        tran_var = parse_result['args']['transcript']
        query = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return summary_var,tran_var, query,  output_var
    
    def execute(self, prog_step, inspect=False):
        summary_var, tran_var, query,  output_var = self.parse(prog_step)
        transcript = prog_step.state[tran_var]
        if isinstance(summary_var, str):
            summary = summary_var
        else:
            summary = prog_step.state[summary_var]

        os.environ["OPENAI_API_KEY"] = "INSERT_KEY_HERE"

        options = prog_step.state['OPTIONS']
        
        opt = f'OPTION A: {options[0]}, OPTION B: {options[1]}, OPTION C: {options[2]}, OPTION D: {options[3]}'

        prompt = f"Given the transcript and its video summary (optional) describing the following context of the video, please answer the following question about the video as accurately as possible.  \
            Only choose from the given options. Just output the option number (i.e. 0 for option A, 1 for option B, 2 for option C and 3 for option D.)\
            \nTranscript: {transcript} \
            \nVideo Summary: {summary} \
            \nQuestion: {query}\
            \nOptions: {opt}"

        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        )

        completion = client.completions.create( # Change the method
            model = "gpt-3.5-turbo-instruct",
            prompt = prompt,
            temperature=0.7,
            max_tokens=512,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1
        )


        # prog = completion.choices[0].text.lstrip('\n').rstrip('\n')
        # return prog, 0
        
        answer = completion.choices[0].text.lstrip('\n').rstrip('\n')
        prog_step.state[output_var] = answer
        return answer


class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def execute(self, prog_step, inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value
        
        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        # print("Step: ",step_input, type(step_input))
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output

class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output

class GetTimeIndexInterpreter():
    step_name = 'GET_TIME_INDEX'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        vid_var = parse_result['args']['video']
        time = eval(parse_result['args']['time'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return vid_var, time, output_var

    def execute(self,prog_step,inspect=False):
        vid_var, time_str, output_var = self.parse(prog_step)
        full_time = prog_step.state["FULL_TIME"]
        x = time.strptime(time_str.split(',')[0],'%M:%S')
        time_sec = datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        ratio = time_sec / full_time
        print("TIME ratio:", f"{time_str} / {full_time} = {time_sec} / {full_time} = {ratio}")
        prog_step.state[output_var] = ratio
        return ratio


def register_step_interpreters(dataset='none'):
    return dict(
        TRACK=TrackInterpreter(),
        CROP=CropInterpreter(),
        TRIM=TrimInterpreter(),
        TRIM_AFTER=TrimAfterInterpreter(),
        TRIM_BEFORE=TrimBeforeInterpreter(),
        MERGE=MergeInterpreter(),
        COUNT=CountInterpreter(),
        SUMMARIZE=SummaryInterpreter(),
        PREDICT=PredictInterpreter(),
        RESULT=ResultInterpreter(),
        GET_TIME_INDEX=GetTimeIndexInterpreter(),
        ANALYSE=AnalyseInterpreter()
    )