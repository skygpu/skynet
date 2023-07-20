import discord
import logging
from skynet.constants import *


class SkynetView(discord.ui.View):

    def __init__(self, bot):
        self.bot = bot
        super().__init__(timeout=None)
        self.add_item(Txt2ImgButton('Txt2Img', discord.ButtonStyle.green, self.bot))
        self.add_item(HelpButton('Help', discord.ButtonStyle.grey))


class Txt2ImgButton(discord.ui.Button):

    def __init__(self, label:str, style:discord.ButtonStyle, bot):
        self.bot = bot
        super().__init__(label=label, style = style)

    async def callback(self, interaction):
        db_call = self.bot.db_call
        work_request = self.bot.work_request
        msg = await grab('Text Prompt:', interaction)
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

        ec = await work_request(user.name, status_msg, 'txt2img', params, msg)

        if ec == 0:
            await db_call('increment_generated', user.id)


class HelpButton(discord.ui.Button):

    def __init__(self, label:str, style:discord.ButtonStyle):
        super().__init__(label=label, style = style)

    async def callback(self, interaction):
        msg = await grab('What would you like help with? (a for all)', interaction)

        param = msg.content

        if param == 'a':
            await msg.reply(content=HELP_TEXT)

        else:
            if param in HELP_TOPICS:
                await msg.reply(content=HELP_TOPICS[param])

            else:
                await msg.reply(content=HELP_UNKWNOWN_PARAM)



async def grab(prompt, interaction):
    def vet(m):
        return m.author == interaction.user and m.channel == interaction.channel

    await interaction.response.send_message(prompt, ephemeral=True)
    message = await interaction.client.wait_for('message', check=vet)
    return message


