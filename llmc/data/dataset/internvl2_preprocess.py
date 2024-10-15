import sys

import torch
sys.path.append('/mnt/afs/fuzuoyi/codes/llms/polysense')
from polys.datasets import build_pipeline
from polys.commons import get_conv_template

image_size = 448
num_image_token=256
max_dynamic_patch=6
img_norm_cfg = dict(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

video_min_num_frames = 3
video_max_num_frames = 6
video_max_dynamic_patch=2

demo_pipeline = [
    dict(type='CopyFilenameToImage'),

    dict(type='GetImageVideoOrder'),

    # img pipeline
    dict(type='ProcessMultiImage',
        pipeline=[
            dict(type='PIL_LoadImageFromFile', 
                file_client_args=dict(backend='petrel'),
            ),
            dict(type='Pil_ToRGBIfNeed'),
            dict(type='DynamicHighResolution',
                using_dynamic_image_size = True,
                min_dynamic_patch=1,
                max_dynamic_patch=max_dynamic_patch,
                image_size=image_size,
                use_thumbnail=True,
            ),
            dict(type='TransformMultiImages',
                pipeline=[
                    dict(type='Pil_ToRGBIfNeed'),
                    dict(type='Pil_Resize', size = image_size, interpolation='InterpolationMode.BICUBIC'),
                    dict(type='Tv_ToTensor'),
                    dict(type='Tv_Normalize', **img_norm_cfg),
                ],
                concat_img=True
            ),
        ], 
    ),

    # video pipeline
    dict(type='ProcessMultiVideo',
        load_pipeline=[
            dict(type='LoadVideoToImageList',
                file_client_args=dict(backend='petrel'),
                min_num_frames=video_min_num_frames,
                max_num_frames=video_max_num_frames,
                sample='rand',
                clip=None,),
        ],
        pipeline=[
            dict(type='Pil_ToRGBIfNeed'),
            dict(type='DynamicHighResolution',
                using_dynamic_image_size = True,
                min_dynamic_patch=1,
                max_dynamic_patch=video_max_dynamic_patch,
                image_size=image_size,
                use_thumbnail=True,
            ),
            dict(type='TransformMultiImages',
                pipeline=[
                    dict(type='Pil_ToRGBIfNeed'),
                    dict(type='Pil_Resize', size = image_size, interpolation='InterpolationMode.BICUBIC'),
                    dict(type='Tv_ToTensor'),
                    dict(type='Tv_Normalize', **img_norm_cfg),
                ],
                concat_img=True
            ),
        ],
    ),

    dict(type='SetImagePlaceholderForVideos'),

    dict(type='MergeImageVideoPixels', image_size=image_size),

    dict(type='Collect',
        keys=['pixel_values', 'img_count', 'img_tiles', 'image_flags', 'text'],
    ),
]


class InternVL2_PreProcess():
    def __init__(self,
                tokenizer=None,
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                IMG_START_TOKEN='<img>', 
                IMG_END_TOKEN='</img>',
                num_image_token=num_image_token,
                max_seq_len=None,
                template='internlm2-chat-v2',
                generation_config=dict()):
        
        self.tokenizer = tokenizer
        self.pipeline = build_pipeline(demo_pipeline)
        
        self.template = template
        # self.template = 'internlm2-chat_cabin_v2'

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.IMG_START_TOKEN = IMG_START_TOKEN
        self.IMG_CONTEXT_TOKEN = IMG_CONTEXT_TOKEN
        self.IMG_END_TOKEN = IMG_END_TOKEN

        self.max_seq_len = max_seq_len
        self.num_image_token = num_image_token

        self.generation_config = generation_config
    
    def __call__(self,
            text= None,
            images = None,
            videos=None,
            audios=None,
            systems=None,
            return_tensors='pt',
            padding=True,
            ):
        # # text is questions or formateed prompt

        # import pdb
        # pdb.set_trace()

        inst_info = dict()

        assert isinstance(text, list)

        if images is not None:
            assert isinstance(images, list)
            inst_info['images'] = len(images)
        else:
            inst_info['images'] = None

        if videos is not None:
            assert isinstance(videos, list)
            inst_info['videos'] = len(videos)
        else:
            inst_info['videos'] = None

        if audios is not None:
            assert isinstance(audios, list)
            inst_info['audios'] = len(audios)
        else:
            inst_info['audios'] = None



        # ******************************************************************
        # parse inst results

        prompts = []
        merged_pixel_values = []
        merged_num_patches_list = []

        for idx in range(len(text)):

            prompt=text[idx]

            results = dict(

            )

            if images is not None:
                image = images[idx]
                if image is not None:
                    results['image'] = image
            else:
                image = None

            if videos is not None:
                video = videos[idx]
                if video is not None:
                    results['video'] = video
            else:
                video = None

            if audios is not None:
                audio = audios[idx]
                if audio is not None:
                    results['audio'] = audio
            else:
                audio = None

            if video is None and image is not None:
                assert '<image>\n' in prompt

            results['text'] = dict(
                    question=prompt,
                    answer='',
            )

            results = self.pipeline(results)

            processed_data_item = results

            pixel_values = processed_data_item['pixel_values']
            img_count = processed_data_item['img_count']
            img_tiles = processed_data_item['img_tiles']
            image_flags = processed_data_item['image_flags']

            if image_flags.numel() == 1 and image_flags.item() == 0:
                # just placeholder
                video_img_inputs = dict(
                    pixel_values=None,
                    num_patches_list=None,
                ) 
            else:
                video_img_inputs = dict(
                    pixel_values=pixel_values,
                    num_patches_list=img_tiles,
                ) 
            
            prompt = processed_data_item['text']['question']
            # Gt_Answer = processed_data_item['text'].get('answer','')
            
            # if video_img_inputs['pixel_values'] is not None and '<image>\n' not in prompt:
            #     new_prompt = '<image>\n{}'.format(prompt)
            #     processed_data_item['text']['question'] = new_prompt

            prompts.append(processed_data_item['text']['question'])
            if video_img_inputs['pixel_values'] is not None:
                merged_pixel_values.append(video_img_inputs['pixel_values'])
                merged_num_patches_list.append(video_img_inputs['num_patches_list'])
            else:
                merged_num_patches_list.append(0)

        if len(merged_pixel_values) == 0:
            merged_pixel_values = None
        else:
            merged_pixel_values = torch.cat(merged_pixel_values, dim=0)

        video_img_inputs = dict(
            pixel_values=merged_pixel_values,
            num_patches_list=merged_num_patches_list,
        )

        # ******************************************************************
        # batch chat 
        pixel_values = video_img_inputs['pixel_values']
        num_patches_list = video_img_inputs['num_patches_list']
        questions = prompts

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                raise NotImplementedError()
            # import pdb
            # pdb.set_trace()

            if '<|im_start|>' in question:
                # question is aleady a formated prompt
                query = question
            else:
                template = get_conv_template(self.template)
                if systems is not None:
                    assert len(systems) == len(questions)
                    system_mess = systems[idx]
                    if system_mess is not None:
                        print('set_system_message: {}'.format(system_mess))
                        template.set_system_message(system_mess)
                template.append_message(template.roles[0], question)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
            
            if not isinstance(num_patches, (list, tuple)):
                num_patches = [num_patches]
            for _num_patches_i in num_patches:
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * _num_patches_i + self.IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            queries.append(query)


        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']


        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0],
            self.tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]

        eos_token_id = [vi for vi in eos_token_id if vi is not None]

        processed_batch_input_datas = dict(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            eos_token_id = eos_token_id,
            **self.generation_config,
        )
        return processed_batch_input_datas


