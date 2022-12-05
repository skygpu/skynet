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
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

from huggingface_hub import login
from datetime import datetime

from pymongo import MongoClient


db_user = os.environ['DB_USER']
db_pass = os.environ['DB_PASS']

logging.basicConfig(level=logging.INFO)

MEM_FRACTION = .33

HELP_TEXT = '''
test art bot v0.1a3

commands work on a user per user basis!
config is individual to each user!

/txt2img {prompt} - request an image based on a prompt

/redo - redo last primpt

/cool - list of cool words to use

/stats - user statistics

/config step {number} - set amount of iterations
/config seed {number} - set the seed, deterministic results!
/config size {width} {height} - set size in pixels
/config guidance {number} - prompt text importance
'''

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

GROUP_ID = -889553587

MIN_STEP = 1
MAX_STEP = 100
MAX_SIZE = (512, 656)
MAX_GUIDANCE = 20

DEFAULT_SIZE = (512, 512)
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEP = 75
DEFAULT_CREDITS = 10

rr_total = 2
rr_id = 0
request_counter = 0

def its_my_turn():
    global request_counter, rr_total, rr_id
    my_turn = request_counter % rr_total == rr_id
    logging.info(f'new request {request_counter}, turn: {my_turn} rr_total: {rr_total}, rr_id {rr_id}')
    request_counter += 1
    return my_turn


def generate_image(i, prompt, name, step, size, guidance, seed):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        revision="fp16",
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
            'config': {
                'step': DEFAULT_STEP,
                'size': DEFAULT_SIZE,
                'seed': None,
                'guidance': DEFAULT_GUIDANCE
            }
        })

        assert res.acknowledged

        return get_user(uid)


    def get_or_create_user(uid: int):
        db_user = get_user(uid)

        if not db_user:
            db_user = new_user(uid)

        return db_user


    def update_user(uid: int, updt_cmd: dict):
        user = get_user(uid)
        if not user:
            raise ValueError('User not present on db')

        return tg_users.find_one_and_update(
            {'uid': uid}, updt_cmd)


    # bot handler
    def img_for_user_with_prompt(
        uid: int,
        prompt: str, step: int, size: tuple[int, int], guidance: int, seed: int
    ):
        name = uuid.uuid4()

        spawn(
            generate_image,
            args=(prompt, name, step, size, guidance, seed))

        logging.info(f'done generating. got {name}, sending...')

        if len(prompt) > 256:
            reply_txt = f'prompt: \"{prompt[:256]}...\"\n(full prompt too big to show on reply...)\n'

        else:
            reply_txt = f'prompt: \"{prompt}\"\n'

        reply_txt +=  f'seed: {seed}\n'
        reply_txt +=  f'iterations: {step}\n'
        reply_txt +=  f'size: {size}\n'
        reply_txt +=  f'guidance: {guidance}\n'
        reply_txt +=  f'stable-diff v1.5 uncensored\n'
        reply_txt +=  f'euler ancestral discrete'

        return reply_txt, name

    @bot.message_handler(commands=['help'])
    def send_help(message):
        if its_my_turn():
            bot.reply_to(message, HELP_TEXT)

    @bot.message_handler(commands=['cool'])
    def send_cool_words(message):
        if its_my_turn():
            bot.reply_to(message, '\n'.join(COOL_WORDS))

    @bot.message_handler(commands=['txt2img'])
    def send_txt2img(message):
        if not its_my_turn():
            return

        # check msg comes from testing group
        chat = message.chat
        if chat.type != 'group' and chat.id != GROUP_ID:
            return

        prompt = ' '.join(message.text.split(' ')[1:])

        if len(prompt) == 0:
            bot.reply_to(message, 'empty text prompt ignored.')
            return

        user = message.from_user
        db_user = get_or_create_user(user.id)

        logging.info(f"{user.first_name} ({user.id}) on chat {chat.id} txt2img: {prompt}")

        user_conf = db_user['config']

        step = user_conf['step']
        size = user_conf['size']
        seed = user_conf['seed'] if user_conf['seed'] else random.randint(0, 999999999)
        guidance = user_conf['guidance']

        try:
            reply_txt, name = img_for_user_with_prompt(
                user.id, prompt, step, size, guidance, seed)

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
    def redo_txt2img(message):
        if not its_my_turn():
            return

        # check msg comes from testing group
        chat = message.chat
        if chat.type != 'group' and chat.id != GROUP_ID:
            return

        user = message.from_user
        db_user = get_or_create_user(user.id)

        prompt = db_user['last_prompt']

        if not prompt:
            bot.reply_to(message, 'do a /txt2img command first silly!')
            return

        user_conf = db_user['config']

        step = user_conf['step']
        size = user_conf['size']
        seed = user_conf['seed'] if user_conf['seed'] else random.randint(0, 999999999)
        guidance = user_conf['guidance']

        logging.info(f"{user.first_name} ({user.id}) on chat {chat.id} redo: {prompt}")

        try:
            reply_txt, name = img_for_user_with_prompt(
                user.id, prompt, step, size, guidance, seed)

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
    def set_config(message):
        if not its_my_turn():
            return

        params = message.text.split(' ')

        if len(params) < 3:
            bot.reply_to(message, 'wrong msg format')

        else:
            user = message.from_user
            chat = message.chat
            db_user = get_user(user.id)

            if not db_user:
                db_user = new_user(user.id)

            try:
                attr = params[1]

                if attr == 'step':
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

                bot.reply_to(message, f"config updated! {attr} to {val}")

            except ValueError:
                bot.reply_to(message, f"\"{val}\" is not a number silly")

    @bot.message_handler(commands=['stats'])
    def user_stats(message):
        if not its_my_turn():
            return

        user = message.from_user
        db_user = get_user(user.id)

        if not db_user:
            db_user = new_user(user.id)

        joined_date_str = datetime.fromisoformat(db_user['joined']).strftime('%B the %dth %Y, %H:%M:%S')

        user_stats_str = f'generated: {db_user["generated"]}\n'
        user_stats_str += f'joined: {joined_date_str}\n'
        user_stats_str += f'credits: {db_user["credits"]}\n'

        bot.reply_to(
            message, user_stats_str)


    login(token=os.environ['HF_TOKEN'])
    bot.infinity_polling()
