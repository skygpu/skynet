#!/usr/bin/python

VERSION = '0.1a12'

DOCKER_RUNTIME_CUDA = 'skynet:runtime-cuda'

MODELS = {
    'prompthero/openjourney':                           {'short': 'midj',        'mem': 6},
    'runwayml/stable-diffusion-v1-5':                   {'short': 'stable',      'mem': 6},
    'stabilityai/stable-diffusion-2-1-base':            {'short': 'stable2',     'mem': 6},
    'snowkidy/stable-diffusion-xl-base-0.9':            {'short': 'stablexl0.9', 'mem': 8.3},
    'Linaqruf/anything-v3.0':                           {'short': 'hdanime',     'mem': 6},
    'hakurei/waifu-diffusion':                          {'short': 'waifu',       'mem': 6},
    'nitrosocke/Ghibli-Diffusion':                      {'short': 'ghibli',      'mem': 6},
    'dallinmackay/Van-Gogh-diffusion':                  {'short': 'van-gogh',    'mem': 6},
    'lambdalabs/sd-pokemon-diffusers':                  {'short': 'pokemon',     'mem': 6},
    'Envvi/Inkpunk-Diffusion':                          {'short': 'ink',         'mem': 6},
    'nousr/robo-diffusion':                             {'short': 'robot',       'mem': 6},
    # Note: not sure about mem
    # 'black-forest-labs/FLUX.1-dev':                     {'short': 'flux',        'mem': 8.3},
    'stabilityai/stable-diffusion-3-medium-diffusers':  {'short': 'stable3', 'mem': 8.3},
    # default is always last
    'stabilityai/stable-diffusion-xl-base-1.0':         {'short': 'stablexl',    'mem': 8.3},

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
/img2img <attach_image> TEXT - request an image base on an image and a prompt

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
/config seed [auto|NUMBER] - set the seed, deterministic results!
/config width NUMBER - set horizontal size in pixels
/config height NUMBER - set vertical size in pixels
/config upscaler [off/x4] - enable/disable x4 size upscaler
/config guidance NUMBER - prompt text importance
/config strength NUMBER - importance of the input image for img2img
'''

UNKNOWN_CMD_TEXT = 'Unknown command! Try sending \"/help\"'

DONATION_INFO = '0xf95335682DF281FFaB7E104EB87B69625d9622B6\ngoal: 0.0465/1.0000 ETH'

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

CLEAN_COOL_WORDS = [
    'cyberpunk',
    'soviet propaganda poster',
    'rastafari',
    'cannabis',
    'art deco',
    'H R Giger Necronom IV',
    'dimethyltryptamine',
    'lysergic',
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
 to a given text input. Value range 0 to 20. Recomended range: 4.5-7.5.
''',

    'strength': '''
Noise is added to the image you use as an init image for img2img, and then the\
 diffusion process continues according to the prompt. The amount of noise added\
 depends on the \"Strength of img2img\"” parameter, which ranges from 0 to 1,\
 where 0 adds no noise at all and you will get the exact image you added, and\
 1 completely replaces the image with noise and almost acts as if you used\
 normal txt2img instead of img2img.
'''
}

HELP_UNKWNOWN_PARAM = 'don\'t have any info on that.'

GROUP_ID = -1001541979235

MP_ENABLED_ROLES = ['god']

MIN_STEP = 1
MAX_STEP = 100
MAX_WIDTH = 1024
MAX_HEIGHT = 1024
MAX_GUIDANCE = 20

DEFAULT_SEED = None
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_GUIDANCE = 7.5
DEFAULT_STRENGTH = 0.5
DEFAULT_STEP = 28
DEFAULT_CREDITS = 10
DEFAULT_MODEL = list(MODELS.keys())[-1]
DEFAULT_ROLE = 'pleb'
DEFAULT_UPSCALER = None

DEFAULT_CONFIG_PATH = 'skynet.toml'

DEFAULT_INITAL_MODELS = [
    'stabilityai/stable-diffusion-xl-base-1.0'
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

DEFAULT_EXPLORER_DOMAIN = 'explorer.skygpu.net'
DEFAULT_IPFS_DOMAIN = 'ipfs.skygpu.net'

DEFAULT_IPFS_REMOTE = '/ip4/169.197.140.154/tcp/4001/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv'
DEFAULT_IPFS_LOCAL = 'http://127.0.0.1:5001'

TG_MAX_WIDTH = 1280
TG_MAX_HEIGHT = 1280

DEFAULT_SINGLE_CARD_MAP = 'cuda:0'
