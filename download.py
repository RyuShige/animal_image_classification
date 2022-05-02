from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#  apiキー情報

key = "5ff88fa35d638a6aa172f77193fae701"
secret = "def35528f4ed5dfb"
wait_time = 1

#  保存フォルダの設定
animal_name = sys.argv[1] #コマンドラインの2つ目の文字を取得
save_dir = "./" + animal_name

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = animal_name,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1, #UIを取得しない？
    extras = 'url_q, licence'# 画像URLを取得
)

#  取得した情報を表示
photos = result['photos']
# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = save_dir + '/' + photo['id'] + '.jpeg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
