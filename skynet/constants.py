#!/usr/bin/python

VERSION = '0.1a10'

DOCKER_RUNTIME_CUDA = 'skynet:runtime-cuda'

MODELS = {
    'prompthero/openjourney':          { 'short': 'midj'},
    'runwayml/stable-diffusion-v1-5':  { 'short': 'stable'},
    'Linaqruf/anything-v3.0':          { 'short': 'hdanime'},
    'hakurei/waifu-diffusion':         { 'short': 'waifu'},
    'nitrosocke/Ghibli-Diffusion':     { 'short': 'ghibli'},
    'dallinmackay/Van-Gogh-diffusion': { 'short': 'van-gogh'},
    'lambdalabs/sd-pokemon-diffusers': { 'short': 'pokemon'},
    'Envvi/Inkpunk-Diffusion':         { 'short': 'ink'},
    'nousr/robo-diffusion':            { 'short': 'robot'}
}

SHORT_NAMES = [
    model_info['short']
    for model_info in MODELS.values()
]

def get_model_by_shortname(short: str):
    for model, info in MODELS.items():
        if short == info['short']:
            return model

N = '\n'
HELP_TEXT = f'''
test art bot v{VERSION}

commands work on a user per user basis!
config is individual to each user!

/txt2img TEXT - request an image based on a prompt

/redo - redo last command (only works for txt2img for now!)

/help step - get info on step config option
/help guidance - get info on guidance config option

/cool - list of cool words to use
/stats - user statistics
/donate - see donation info

/config algo NAME - select AI to use one of:
/config model NAME - select AI to use one of:

{N.join(SHORT_NAMES)}

/config step NUMBER - set amount of iterations
/config seed NUMBER - set the seed, deterministic results!
/config size WIDTH HEIGHT - set size in pixels
/config guidance NUMBER - prompt text importance
'''

UNKNOWN_CMD_TEXT = 'Unknown command! Try sending \"/help\"'

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

HELP_TOPICS = {
    'step': '''
Diffusion models are iterative processes – a repeated cycle that starts with a\
 random noise generated from text input. With each step, some noise is removed\
, resulting in a higher-quality image over time. The repetition stops when the\
 desired number of steps completes.

Around 25 sampling steps are usually enough to achieve high-quality images. Us\
ing more may produce a slightly different picture, but not necessarily better \
quality.
''',

'guidance': '''
The guidance scale is a parameter that controls how much the image generation\
 process follows the text prompt. The higher the value, the more image sticks\
 to a given text input.
'''
}

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
DEFAULT_STRENGTH = 0.5
DEFAULT_STEP = 35
DEFAULT_CREDITS = 10
DEFAULT_MODEL = list(MODELS.keys())[0]
DEFAULT_ROLE = 'pleb'
DEFAULT_UPSCALER = None

DEFAULT_CONFIG_PATH = 'skynet.ini'

DEFAULT_INITAL_MODELS = [
    'prompthero/openjourney',
    'runwayml/stable-diffusion-v1-5'
]

DATE_FORMAT = '%B the %dth %Y, %H:%M:%S'

CONFIG_ATTRS = [
    'algo',
    'step',
    'width',
    'height',
    'seed',
    'guidance',
    'strength',
    'upscaler'
]

DEFAULT_IPFS_REMOTE = '/ip4/169.197.140.154/tcp/4001/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv'
