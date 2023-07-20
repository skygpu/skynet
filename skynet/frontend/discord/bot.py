# import os
import discord
# import asyncio
# from dotenv import load_dotenv
# from pathlib import Path
from discord.ext import commands
from .ui import SkynetView


# # Auth
# current_dir = Path(__file__).resolve().parent
# # parent_dir = current_dir.parent
# env_file_path = current_dir / ".env"
# load_dotenv(dotenv_path=env_file_path)
#
# discordToken = os.getenv("DISCORD_TOKEN")


# Actual Discord bot.
class DiscordBot(commands.Bot):

    def __init__(self, bot, *args, **kwargs):
        self.bot = bot
        intents = discord.Intents(
            messages=True,
            guilds=True,
            typing=True,
            members=True,
            presences=True,
            reactions=True,
            message_content=True,
            voice_states=True
        )
        super().__init__(command_prefix='/', intents=intents, *args, **kwargs)

    # async def setup_hook(self):
    #     db.poll_db.start()

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord!')
        for guild in self.guilds:
            for channel in guild.channels:
                if channel.name == "skynet":
                    await channel.send('Skynet bot online', view=SkynetView(self.bot))

        print("\n==============")
        print("Logged in as")
        print(self.user.name)
        print(self.user.id)
        print("==============")

    async def on_message(self, message):
        if isinstance(message.channel, discord.DMChannel):
            return
        elif message.channel.name != 'skynet':
            return
        elif message.author != self.user:
            await self.process_commands(message)
        await message.channel.send('', view=SkynetView(self.bot))

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send('You missed a required argument, please try again.')

    # async def on_message(self, message):
    #     print(f"message from {message.author} what he said {message.content}")
        # await message.channel.send(message.content)

# bot=DiscordBot()
# @bot.command(name='config', help='Responds with the configuration')
# async def config(ctx):
#     response = "This is the bot configuration"  # Put your bot configuration here
#     await ctx.send(response)
#
# @bot.command(name='helper', help='Responds with a help')
# async def helper(ctx):
#     response = "This is help information" # Put your help response here
#     await ctx.send(response)
#
# @bot.command(name='txt2img', help='Responds with an image')
# async def txt2img(ctx, *, arg):
#     response = f"This is your prompt: {arg}"
#     await ctx.send(response)
# bot.run(discordToken)
