import discord
import logging
from skynet.constants import *
from skynet.frontend import validate_user_config_request


class SkynetView(discord.ui.View):

    def __init__(self, bot):
        self.bot = bot
        super().__init__(timeout=None)
        self.add_item(RedoButton('redo', discord.ButtonStyle.green, self.bot))
        self.add_item(Txt2ImgButton('txt2img', discord.ButtonStyle.green, self.bot))
        self.add_item(ConfigButton('config', discord.ButtonStyle.grey, self.bot))
        self.add_item(HelpButton('help', discord.ButtonStyle.grey, self.bot))
        self.add_item(CoolButton('cool', discord.ButtonStyle.gray, self.bot))


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
        status_msg = await msg.reply(init_msg)
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

        if ec == 0:
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
            await interaction.response.edit_message(
                'no last prompt found, do a txt2img cmd first!'
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
        await work_request(
            user, status_msg, 'redo', params, interaction,
            file_id=file_id,
            binary_data=binary
        )


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


class CoolButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        await interaction.response.send_message(
            content='\n'.join(CLEAN_COOL_WORDS),
            view=SkynetView(self.bot))


class HelpButton(discord.ui.Button):

    def __init__(self, label: str, style: discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style=style)

    async def callback(self, interaction):
        msg = await grab('What would you like help with? (a for all)', interaction)

        param = msg.content

        if param == 'a':
            await msg.reply(content=HELP_TEXT, view=SkynetView(self.bot))

        else:
            if param in HELP_TOPICS:
                await msg.reply(content=HELP_TOPICS[param], view=SkynetView(self.bot))

            else:
                await msg.reply(content=HELP_UNKWNOWN_PARAM, view=SkynetView(self.bot))



async def grab(prompt, interaction):
    def vet(m):
        return m.author == interaction.user and m.channel == interaction.channel

    await interaction.response.send_message(prompt, ephemeral=True)
    message = await interaction.client.wait_for('message', check=vet)
    return message


