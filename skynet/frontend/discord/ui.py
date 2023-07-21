import io
import discord
from PIL import Image
import logging
from skynet.constants import *
from skynet.frontend import validate_user_config_request


class SkynetView(discord.ui.View):

    def __init__(self, bot):
        self.bot = bot
        super().__init__(timeout=None)
        self.add_item(RedoButton('redo', discord.ButtonStyle.primary, self.bot))
        self.add_item(Txt2ImgButton('txt2img', discord.ButtonStyle.primary, self.bot))
        self.add_item(Img2ImgButton('img2img', discord.ButtonStyle.primary, self.bot))
        self.add_item(StatsButton('stats', discord.ButtonStyle.secondary, self.bot))
        self.add_item(DonateButton('donate', discord.ButtonStyle.secondary, self.bot))
        self.add_item(ConfigButton('config', discord.ButtonStyle.secondary, self.bot))
        self.add_item(HelpButton('help', discord.ButtonStyle.secondary, self.bot))
        self.add_item(CoolButton('cool', discord.ButtonStyle.secondary, self.bot))


class Txt2ImgButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        db_call = self.bot.db_call
        work_request = self.bot.work_request
        msg = await grab('Enter your prompt:', interaction)
        # grab user from msg
        user = msg.author
        user_row = await db_call('get_or_create_user', user.id)

        # init new msg
        init_msg = 'started processing txt2img request...'
        status_msg = await msg.channel.send(init_msg)
        await db_call(
            'new_user_request', user.id, msg.id, status_msg.id, status=init_msg)

        prompt = msg.content

        if len(prompt) == 0:
            await status_msg.edit(content=
                'Empty text prompt ignored.'
            )
            await db_call('update_user_request', status_msg.id, 'Empty text prompt ignored.')
            return

        logging.info(f'mid: {msg.id}')

        user_config = {**user_row}
        del user_config['id']

        params = {
            'prompt': prompt,
            **user_config
        }

        await db_call(
            'update_user_stats', user.id, 'txt2img', last_prompt=prompt)

        ec = await work_request(user, status_msg, 'txt2img', params, msg)

        if ec == None:
            await db_call('increment_generated', user.id)


class Img2ImgButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        db_call = self.bot.db_call
        work_request = self.bot.work_request
        ipfs_node = self.bot.ipfs_node
        msg = await grab('Attach an Image. Enter your prompt:', interaction)

        user = msg.author
        user_row = await db_call('get_or_create_user', user.id)

        # init new msg
        init_msg = 'started processing img2img request...'
        status_msg = await msg.channel.send(init_msg)
        await db_call(
            'new_user_request', user.id, msg.id, status_msg.id, status=init_msg)

        # if not msg.content.startswith('/img2img'):
        #     await msg.reply(
        #         'For image to image you need to add /img2img to the beggining of your caption'
        #     )
        #     return

        prompt = msg.content

        if len(prompt) == 0:
            await msg.reply('Empty text prompt ignored.')
            return

        # file_id = message.photo[-1].file_id
        # file_path = (await bot.get_file(file_id)).file_path
        # image_raw = await bot.download_file(file_path)
        #

        file = msg.attachments[-1]
        file_id = str(file.id)
        # file bytes
        image_raw = await file.read()
        with Image.open(io.BytesIO(image_raw)) as image:
            w, h = image.size

            if w > 512 or h > 512:
                logging.warning(f'user sent img of size {image.size}')
                image.thumbnail((512, 512))
                logging.warning(f'resized it to {image.size}')

            image.save(f'ipfs-docker-staging/image.png', format='PNG')

            ipfs_hash = ipfs_node.add('image.png')
            ipfs_node.pin(ipfs_hash)

            logging.info(f'published input image {ipfs_hash} on ipfs')

        logging.info(f'mid: {msg.id}')

        user_config = {**user_row}
        del user_config['id']

        params = {
            'prompt': prompt,
            **user_config
        }

        await db_call(
            'update_user_stats',
            user.id,
            'img2img',
            last_file=file_id,
            last_prompt=prompt,
            last_binary=ipfs_hash
        )

        ec = await work_request(
            user, status_msg, 'img2img', params, msg,
            file_id=file_id,
            binary_data=ipfs_hash
        )

        if ec == None:
            await db_call('increment_generated', user.id)


class RedoButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        db_call = self.bot.db_call
        work_request = self.bot.work_request
        init_msg = 'started processing redo request...'
        await interaction.response.send_message(init_msg)
        status_msg = await interaction.original_response()
        user = interaction.user

        method = await db_call('get_last_method_of', user.id)
        prompt = await db_call('get_last_prompt_of', user.id)

        file_id = None
        binary = ''
        if method == 'img2img':
            file_id = await db_call('get_last_file_of', user.id)
            binary = await db_call('get_last_binary_of', user.id)

        if not prompt:
            await status_msg.edit(
                content='no last prompt found, do a txt2img cmd first!',
                view=SkynetView(self.bot)
            )
            return

        user_row = await db_call('get_or_create_user', user.id)
        await db_call(
            'new_user_request', user.id, interaction.id, status_msg.id, status=init_msg)
        user_config = {**user_row}
        del user_config['id']

        params = {
            'prompt': prompt,
            **user_config
        }
        ec = await work_request(
            user, status_msg, 'redo', params, interaction,
            file_id=file_id,
            binary_data=binary
        )

        if ec == None:
            await db_call('increment_generated', user.id)


class ConfigButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        db_call = self.bot.db_call
        msg = await grab('What params do you want to change? (format: <param> <value>)', interaction)

        user = interaction.user
        try:
            attr, val, reply_txt = validate_user_config_request(
                '/config ' + msg.content)

            logging.info(f'user config update: {attr} to {val}')
            await db_call('update_user_config', user.id, attr, val)
            logging.info('done')

        except BaseException as e:
            reply_txt = str(e)

        finally:
            await msg.reply(content=reply_txt, view=SkynetView(self.bot))


class StatsButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        db_call = self.bot.db_call

        user = interaction.user

        await db_call('get_or_create_user', user.id)
        generated, joined, role = await db_call('get_user_stats', user.id)

        stats_str = f'```generated: {generated}\n'
        stats_str += f'joined: {joined}\n'
        stats_str += f'role: {role}\n```'

        await interaction.response.send_message(
            content=stats_str, view=SkynetView(self.bot))


class DonateButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        await interaction.response.send_message(
            content=f'```\n{DONATION_INFO}```',
            view=SkynetView(self.bot))


class CoolButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        clean_cool_word = '\n'.join(CLEAN_COOL_WORDS)
        await interaction.response.send_message(
            content=f'```{clean_cool_word}```',
            view=SkynetView(self.bot))


class HelpButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        msg = await grab('What would you like help with? (a for all)', interaction)

        param = msg.content

        if param == 'a':
            await msg.reply(content=f'```{HELP_TEXT}```', view=SkynetView(self.bot))

        else:
            if param in HELP_TOPICS:
                await msg.reply(content=f'```{HELP_TOPICS[param]}```', view=SkynetView(self.bot))

            else:
                await msg.reply(content=f'```{HELP_UNKWNOWN_PARAM}```', view=SkynetView(self.bot))


async def grab(prompt, interaction):
    def vet(m):
        return m.author == interaction.user and m.channel == interaction.channel

    await interaction.response.send_message(prompt, ephemeral=True)
    message = await interaction.client.wait_for('message', check=vet)
    return message


