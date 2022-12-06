#!/usr/bin/python

import os

import logging
import random

from torch.multiprocessing import spawn

import telebot
from telebot.types import InputFile

import sys
import uuid

from pathlib import Path

import torch
from torch.multiprocessing.spawn import ProcessRaisedException
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler
)

from huggingface_hub import login
from datetime import datetime

from pymongo import MongoClient

from typing import Tuple, Optional

db_user = os.environ['DB_USER']
db_pass = os.environ['DB_PASS']

logging.basicConfig(level=logging.INFO)

MEM_FRACTION = .33

ALGOS = {
    'stable': 'runwayml/stable-diffusion-v1-5',
    'midj': 'prompthero/openjourney',
    'hdanime': 'Linaqruf/anything-v3.0',
    'waifu': 'hakurei/waifu-diffusion',
    'ghibli': 'nitrosocke/Ghibli-Diffusion',
    'van-gogh': 'dallinmackay/Van-Gogh-diffusion',
    'pokemon': 'lambdalabs/sd-pokemon-diffusers',
    'ink': 'Envvi/Inkpunk-Diffusion',
    'robot': 'nousr/robo-diffusion'
}

N = '\n'
HELP_TEXT = f'''
test art bot v0.1a4

commands work on a user per user basis!
config is individual to each user!

/txt2img TEXT - request an image based on a prompt

/redo - redo last prompt

/cool - list of cool words to use
/stats - user statistics
/donate - see donation info

/config algo NAME - select AI to use one of:

{N.join(ALGOS.keys())}

/config step NUMBER - set amount of iterations
/config seed NUMBER - set the seed, deterministic results!
/config size WIDTH HEIGHT - set size in pixels
/config guidance NUMBER - prompt text importance
'''

UNKNOWN_CMD_TEXT = 'unknown command! try sending \"/help\"'

DONATION_INFO = '0xf95335682DF281FFaB7E104EB87B69625d9622B6\ngoal: 25/650usd'

COOL_WORDS = [
    'cyberpunk',
    'soviet propaganda poster',
    'rastafari',
    'cannabis',
    'art deco',
    'H R Giger Necronom IV',
    'dimethyltryptamine',
    'lysergic',
    'slut',
    'psilocybin',
    'trippy',
    'lucy in the sky with diamonds',
    'fractal',
    'da vinci',
    'pencil illustration',
    'blueprint',
    'internal diagram',
    'baroque',
    'the last judgment',
    'michelangelo'
]

GROUP_ID = -1001541979235

MP_ENABLED_ROLES = ['god']

MIN_STEP = 1
MAX_STEP = 100
MAX_SIZE = (512, 656)
MAX_GUIDANCE = 20

DEFAULT_SIZE = (512, 512)
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEP = 75
DEFAULT_CREDITS = 10
DEFAULT_ALGO = 'stable'
DEFAULT_ROLE = 'pleb'
DEFAULT_UPSCALER = None

rr_total = 1
rr_id = 0
request_counter = 0

def its_my_turn():
    global request_counter, rr_total, rr_id
    my_turn = request_counter % rr_total == rr_id
    logging.info(f'new request {request_counter}, turn: {my_turn} rr_total: {rr_total}, rr_id {rr_id}')
    request_counter += 1
    return my_turn

def round_robined(func):
    def rr_wrapper(*args, **kwargs):
        if not its_my_turn():
            return

        func(*args, **kwargs)

    return rr_wrapper


def generate_image(
    i: int,
    prompt: str,
    name: str,
    step: int,
    size: Tuple[int, int],
    guidance: int,
    seed: int,
    algo: str,
    upscaler: Optional[str]
):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    with torch.no_grad():
        if algo == 'stable':
            pipe = StableDiffusionPipeline.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                torch_dtype=torch.float16,
                revision="fp16",
                safety_checker=None
            )

        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                ALGOS[algo],
                torch_dtype=torch.float16,
                safety_checker=None
            )

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        w, h = size
        print(f'generating image... of size {w, h}')
        image = pipe(
            prompt,
            width=w,
            height=h,
            guidance_scale=guidance, num_inference_steps=step,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]

        if upscaler == 'x4':
            pipe = StableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-x4-upscaler',
                revision="fp16",
                torch_dtype=torch.float16
            )
            image = pipe(prompt=prompt, image=image).images[0]


    image.save(f'/outputs/{name}.png')
    print('saved')


if __name__ == '__main__':

    API_TOKEN = '5880619053:AAFge2UfObw1kCn9Kb7AAyqaIHs_HgM0Fx0'

    bot = telebot.TeleBot(API_TOKEN)
    db_client = MongoClient(
        host=['ancap.tech:64000'],
        username=db_user,
        password=db_pass)

    tgdb = db_client.get_database('telegram')

    collections = tgdb.list_collection_names()

    if 'users' in collections:
        tg_users = tgdb.get_collection('users')
        # tg_users.delete_many({})

    else:
        tg_users = tgdb.create_collection('users')

    # db functions

    def get_user(uid: int):
        return tg_users.find_one({'uid': uid})


    def new_user(uid: int):
        if get_user(uid):
            raise ValueError('User already present on db')

        res = tg_users.insert_one({
            'generated': 0,
            'uid': uid,
            'credits': DEFAULT_CREDITS,
            'joined': datetime.utcnow().isoformat(),
            'last_prompt': None,
            'role': DEFAULT_ROLE,
            'config': {
                'algo': DEFAULT_ALGO,
                'step': DEFAULT_STEP,
                'size': DEFAULT_SIZE,
                'seed': None,
                'guidance': DEFAULT_GUIDANCE,
                'upscaler': DEFAULT_UPSCALER
            }
        })

        assert res.acknowledged

        return get_user(uid)

    def migrate_user(db_user):
        # new: user roles
        if 'role' not in db_user:
            res = tg_users.find_one_and_update(
                {'uid': db_user['uid']}, {'$set': {'role': DEFAULT_ROLE}})

        # new: algo selection
        if 'algo' not in db_user['config']:
            res = tg_users.find_one_and_update(
                {'uid': db_user['uid']}, {'$set': {'config.algo': DEFAULT_ALGO}})

        # new: upscaler selection
        if 'upscaler' not in db_user['config']:
            res = tg_users.find_one_and_update(
                {'uid': db_user['uid']}, {'$set': {'config.upscaler': DEFAULT_UPSCALER}})

        return get_user(db_user['uid'])

    def get_or_create_user(uid: int):
        db_user = get_user(uid)

        if not db_user:
            db_user = new_user(uid)

        logging.info(f'req from: {uid}')

        return migrate_user(db_user)

    def update_user(uid: int, updt_cmd: dict):
        user = get_user(uid)
        if not user:
            raise ValueError('User not present on db')

        return tg_users.find_one_and_update(
            {'uid': uid}, updt_cmd)


    # bot handler
    def img_for_user_with_prompt(
        uid: int,
        prompt: str, step: int, size: Tuple[int, int], guidance: int, seed: int,
        algo: str, upscaler: Optional[str]
    ):
        name = uuid.uuid4()

        spawn(
            generate_image,
            args=(prompt, name, step, size, guidance, seed, algo, upscaler))

        logging.info(f'done generating. got {name}, sending...')

        if len(prompt) > 256:
            reply_txt = f'prompt: \"{prompt[:256]}...\"\n(full prompt too big to show on reply...)\n'

        else:
            reply_txt = f'prompt: \"{prompt}\"\n'

        reply_txt +=  f'seed: {seed}\n'
        reply_txt +=  f'iterations: {step}\n'
        reply_txt +=  f'size: {size}\n'
        reply_txt +=  f'guidance: {guidance}\n'
        reply_txt +=  f'algo: {ALGOS[algo]}\n'
        reply_txt +=  f'euler ancestral discrete'

        return reply_txt, name

    @bot.message_handler(commands=['help'])
    @round_robined
    def send_help(message):
        bot.reply_to(message, HELP_TEXT)

    @bot.message_handler(commands=['cool'])
    @round_robined
    def send_cool_words(message):
        bot.reply_to(message, '\n'.join(COOL_WORDS))

    @bot.message_handler(commands=['txt2img'])
    @round_robined
    def send_txt2img(message):
        chat = message.chat
        user = message.from_user
        db_user = get_or_create_user(user.id)

        if ((chat.type != 'group' and chat.id != GROUP_ID) and
                (db_user['role'] not in MP_ENABLED_ROLES)):
            return

        prompt = ' '.join(message.text.split(' ')[1:])

        if len(prompt) == 0:
            bot.reply_to(message, 'empty text prompt ignored.')
            return

        logging.info(f"{user.first_name} ({user.id}) on chat {chat.id} txt2img: {prompt}")

        user_conf = db_user['config']

        algo = user_conf['algo']
        step = user_conf['step']
        size = user_conf['size']
        seed = user_conf['seed'] if user_conf['seed'] else random.randint(0, 999999999)
        guidance = user_conf['guidance']
        upscaler = user_conf['upscaler']

        try:
            reply_txt, name = img_for_user_with_prompt(
                user.id, prompt, step, size, guidance, seed, algo, upscaler)

            update_user(
                user.id,
                {'$set': {
                    'generated': db_user['generated'] + 1,
                    'last_prompt': prompt
                    }})

            bot.send_photo(
                chat.id,
                caption=f'sent by: {user.first_name}\n' + reply_txt,
                photo=InputFile(f'/outputs/{name}.png'))

        except BaseException as e:
            logging.error(e)
            bot.reply_to(message, 'that command caused an error :(\nchange settings and try again (?')

    @bot.message_handler(commands=['redo'])
    @round_robined
    def redo_txt2img(message):
        # check msg comes from testing group
        chat = message.chat
        user = message.from_user
        db_user = get_or_create_user(user.id)

        if ((chat.type != 'group' and chat.id != GROUP_ID) and
                (db_user['role'] not in MP_ENABLED_ROLES)):
            return

        prompt = db_user['last_prompt']

        if not prompt:
            bot.reply_to(message, 'do a /txt2img command first silly!')
            return

        user_conf = db_user['config']

        algo = user_conf['algo']
        step = user_conf['step']
        size = user_conf['size']
        seed = user_conf['seed'] if user_conf['seed'] else random.randint(0, 999999999)
        guidance = user_conf['guidance']
        upscaler = user_conf['upscaler']

        logging.info(f"{user.first_name} ({user.id}) on chat {chat.id} redo: {prompt}")

        try:
            reply_txt, name = img_for_user_with_prompt(
                user.id, prompt, step, size, guidance, seed, algo, upscaler)

            update_user(
                user.id,
                {'$set': {
                    'generated': db_user['generated'] + 1,
                    }})

            bot.send_photo(
                chat.id,
                caption=f'sent by: {user.first_name}\n' + reply_txt,
                photo=InputFile(f'/outputs/{name}.png'))

        except BaseException as e:
            logging.error(e)
            bot.reply_to(message, 'that command caused an error :(\nchange settings and try again (?')

    @bot.message_handler(commands=['config'])
    @round_robined
    def set_config(message):
        logging.info(f'config req on chat: {message.chat.id}')

        params = message.text.split(' ')

        if len(params) < 3:
            bot.reply_to(message, 'wrong msg format')

        else:
            user = message.from_user
            chat = message.chat
            db_user = get_or_create_user(user.id)

            try:
                attr = params[1]

                if attr == 'algo':
                    val = params[2]
                    assert val in ALGOS
                    res = update_user(user.id, {'$set': {'config.algo': val}})

                elif attr == 'step':
                    val = int(params[2])
                    val = max(min(val, MAX_STEP), MIN_STEP)
                    res = update_user(user.id, {'$set': {'config.step': val}})

                elif attr  == 'size':
                    max_w, max_h = MAX_SIZE
                    w = max(min(int(params[2]), max_w), 16)
                    h = max(min(int(params[3]), max_h), 16)

                    val = (w, h)

                    if (w % 8 != 0) or (h % 8 != 0):
                        bot.reply_to(message, 'size must be divisible by 8!')
                        return

                    res = update_user(user.id, {'$set': {'config.size': val}})

                elif attr == 'seed':
                    val = params[2]
                    if val == 'auto':
                        val = None
                    else:
                        val = int(params[2])

                    res = update_user(user.id, {'$set': {'config.seed': val}})

                elif attr == 'guidance':
                    val = float(params[2])
                    val = max(min(val, MAX_GUIDANCE), 0)
                    res = update_user(user.id, {'$set': {'config.guidance': val}})

                elif attr == 'upscaler':
                    val = params[2]
                    if val == 'off':
                        val = None

                    res = update_user(user.id, {'$set': {'config.upscaler': val}})

                else:
                    bot.reply_to(message, f'\"{attr}\" not a parameter')

                bot.reply_to(message, f'config updated! {attr} to {val}')

            except ValueError:
                bot.reply_to(message, f'\"{val}\" is not a number silly')

            except AssertionError:
                bot.reply_to(message, f'no algo named {val}')

    @bot.message_handler(commands=['stats'])
    @round_robined
    def user_stats(message):
        user = message.from_user
        db_user = get_or_create_user(user.id)
        migrate_user(db_user)

        joined_date_str = datetime.fromisoformat(db_user['joined']).strftime('%B the %dth %Y, %H:%M:%S')

        user_stats_str = f'generated: {db_user["generated"]}\n'
        user_stats_str += f'joined: {joined_date_str}\n'
        user_stats_str += f'credits: {db_user["credits"]}\n'
        user_stats_str += f'role: {db_user["role"]}\n'

        bot.reply_to(
            message, user_stats_str)

    @bot.message_handler(commands=['donate'])
    @round_robined
    def donation_info(message):
        bot.reply_to(
            message, DONATION_INFO)

    @bot.message_handler(commands=['say'])
    @round_robined
    def say(message):
        chat = message.chat
        user = message.from_user
        db_user = get_or_create_user(user.id)

        if (chat.type == 'group') or (db_user['role'] not in MP_ENABLED_ROLES):
            return

        bot.send_message(GROUP_ID, message.text[4:])

    @bot.message_handler(func=lambda message: True)
    @round_robined
    def echo_message(message):
        if message.text[0] == '/':
            bot.reply_to(message, UNKNOWN_CMD_TEXT)


    login(token=os.environ['HF_TOKEN'])

    bot.infinity_polling()
