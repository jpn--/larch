
try:
	from selenium import webdriver
	from PIL import Image, ImageDraw, ImageEnhance, ImageChops
except ImportError:
	pass # Nothing in this module will work

import os
import base64
from io import BytesIO



class _ScreenCapper:

	def __init__(self):
		options = webdriver.ChromeOptions()
		options.add_argument('headless')
		options.add_argument('window-size=1024x768')
		options.add_argument("disable-overlay-scrollbar")
		self.driver = webdriver.Chrome(chrome_options=options)

	def __del__(self):
		self.driver.close()

	def _screenshot_formatter(self, url, window_size=(1024, 768), full_height=False):
		self.driver.get(url)
		if window_size[1] is None or full_height:
			# Full height window
			self.driver.maximize_window()
			height = self.driver.execute_script("return document.body.scrollHeight")
			window_size = window_size[0], height
			self.driver.set_window_size(*window_size)

	def screenshot_to_file(self, url, file_name, log=None, output_dir='./', window_size=(1024, 768), full_height=False):
		if log is not None:
			log("Capturing", url, "screenshot as", file_name, "...")
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		self._screenshot_formatter(url, window_size, full_height=full_height)
		self.driver.save_screenshot(file_name)
		if log is not None:
			log("Done capturing", url)

	def screenshot_to_base64(self, url, log=None, window_size=(1024, 768), full_height=False):
		if log is not None:
			log("Capturing", url, "screenshot as base64...")
		self._screenshot_formatter(url, window_size, full_height=full_height)
		result = self.driver.get_screenshot_as_base64()
		if log is not None:
			log("Done capturing", url)
		return result

_grabber = None


def screenshot(url, file_name=None, log=None, output_dir='./', window_size=(1024, 768), full_height=False):
	global _grabber
	if _grabber is None:
		_grabber = _ScreenCapper()
	if file_name is None:
		return _grabber.screenshot_to_base64(url, log=log, window_size=window_size, full_height=full_height)
	else:
		_grabber.screenshot_to_file(url, file_name, log=log, output_dir=output_dir, window_size=window_size, full_height=full_height)

# https://gist.github.com/rinchik/d023578a705d6d4a5b12e235c5a9df9a#file-averageregionbrightness-py

def _process_region(image, x, y, width, height, tolerance=100):
	region_total = 0

	# This can be used as the sensitivity factor, the larger it is the less sensitive the comparison
	factor = tolerance

	for coordinateY in range(y, y + height):
		for coordinateX in range(x, x + width):
			try:
				pixel = image.getpixel((coordinateX, coordinateY))
				region_total += sum(pixel) / 4
			except:
				return

	return region_total / factor


def diff_images(imagefile1, imagefile2, return_blocked1=False, tolerance=100):
	'''

	Parameters
	----------
	imagefile1, imagefile2 : images
		Two images to compare.  They should be the same size.  Give by filename or with
		raw data.
	return_blocked1 : bool, default False
		If true, also return an image `Elem` based on imagefile1 showing where the mismatch blocks are.
	tolerance : int
		This can be used as the sensitivity factor, the larger it is the less sensitive the comparison

	Returns
	-------
	int
		The number of mismatched blocks
	Elem
		optional, see return_blocked1
	'''
	screenshot_staging = Image.open(imagefile1)
	screenshot_production = Image.open(imagefile2)
	columns = 60
	rows = 80
	screen_width, screen_height = screenshot_staging.size

	block_width = ((screen_width - 1) // columns) + 1  # this is just a division ceiling
	block_height = ((screen_height - 1) // rows) + 1

	mismatch_blocks = 0

	if return_blocked1:
		degenerate = Image.new(screenshot_staging.mode, screenshot_staging.size, '#FFFFFF')

		if 'A' in screenshot_staging.getbands():
			degenerate.putalpha(screenshot_staging.getchannel('A'))

		screenshot_staging_1 = Image.blend(degenerate, screenshot_staging, 0.25)



	for y in range(0, screen_height, block_height + 1):
		for x in range(0, screen_width, block_width + 1):
			region_staging = _process_region(screenshot_staging, x, y, block_width, block_height, tolerance=tolerance)
			region_production = _process_region(screenshot_production, x, y, block_width, block_height, tolerance=tolerance)

			if region_staging is not None and region_production is not None and region_production != region_staging:
				mismatch_blocks += 1
				if return_blocked1:
					draw = ImageDraw.Draw(screenshot_staging_1)
					draw.rectangle((x, y, x + block_width, y + block_height), outline="red")

	if return_blocked1:
		buffered = BytesIO()
		screenshot_staging_1.save(buffered, format="PNG")
		from xmle import Elem
		blocked1 = Elem.from_png_raw(buffered.getvalue())
		return mismatch_blocks, blocked1

	return mismatch_blocks




# https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil/10616717

def trim_uniform_border(im):
	"""
	Trim a border from

	Parameters
	----------
	im : PIL.Image.Image

	Returns
	-------
	PIL.Image.Image

	"""
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)
