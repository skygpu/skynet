#!/usr/bin/python

API_TOKEN = '5880619053:AAFge2UfObw1kCn9Kb7AAyqaIHs_HgM0Fx0'

DB_HOST = 'ancap.tech:34508'

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

/redo - re ont

/help step - get info on step config option
/help guidance - get info on guidance config option

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

HELP_STEP = '''
diffusion models are iterative processes – a repeated cycle that starts with a\
 random noise generated from text input. With each step, some noise is removed\
, resulting in a higher-quality image over time. The repetition stops when the\
 desired number of steps completes.

around 25 sampling steps are usually enough to achieve high-quality images. Us\
ing more may produce a slightly different picture, but not necessarily better \
quality.
'''

HELP_GUIDANCE = '''
the guidance scale is a parameter that controls how much the image generation\
 process follows the text prompt. The higher the value, the more image sticks\
 to a given text input.
'''

HELP_UNKWNOWN_PARAM = 'don\'t have any info on that.'

GROUP_ID = -1001541979235

MP_ENABLED_ROLES = ['god']

MIN_STEP = 1
MAX_STEP = 100
MAX_WIDTH = 512
MAX_HEIGHT = 656
MAX_GUIDANCE = 20

DEFAULT_SEED = None
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEP = 35
DEFAULT_CREDITS = 10
DEFAULT_ALGO = 'midj'
DEFAULT_ROLE = 'pleb'
DEFAULT_UPSCALER = None

DEFAULT_RPC_ADDR = 'tcp://127.0.0.1:41000'

DEFAULT_DGPU_ADDR = 'tcp://127.0.0.1:41069'
DEFAULT_DGPU_MAX_TASKS = 3
DEFAULT_INITAL_ALGOS = ['midj', 'stable', 'ink']

DATE_FORMAT = '%B the %dth %Y, %H:%M:%S'

CONFIG_ATTRS = [
    'algo',
    'step',
    'width',
    'height',
    'seed',
    'guidance',
    'upscaler'
]
