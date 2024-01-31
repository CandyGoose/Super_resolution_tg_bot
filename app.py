import os
from generator import generate
import logging
from aiogram import Bot, Dispatcher, executor, types

API_TOKEN: str = os.environ.get('TG_BOT_TOKEN')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    logger.info("Received /start or /help command from user %s", message.from_user.id)
    await message.answer(text="Привет!\n"
                        "Отправь мне фото и я увеличу ее размер.")

@dp.message_handler(commands=['upscale_image'])
async def transfer_style_command(message: types.Message):
    logger.info("Received /upscale_image command from user %s", message.from_user.id)
    with open('data/panda.jpg', 'rb') as photo:
        await message.answer_photo(photo, caption='Тестовая картинка')
    generate('data/panda.jpg').save('data/output.jpg')
    with open('data/output.jpg', 'rb') as photo:
        await message.answer_photo(photo, caption='Увеличенная картинка')
    logger.info("Sent upscaled image to user %s", message.from_user.id)

@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    logger.info("Received a photo from user %s", message.from_user.id)
    await message.photo[-1].download(destination_file="data/input.jpg")
    generate('data/input.jpg').save('data/output.jpg')
    with open('data/output.jpg', 'rb') as photo:
        await message.answer_photo(photo, caption='Увеличенная картинка')
    logger.info("Sent upscaled image in response to photo from user %s", message.from_user.id)

@dp.message_handler()
async def exception(message: types.Message):
    await message.answer("Привет! Нажми /start, чтобы начать.")


if __name__ == '__main__':
    try:
        logger.info("Starting the bot...")
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logger.exception("Error occurred while running the bot.")
    finally:
        logger.info("Stopping the bot...")
