import concurrent.futures
import os
import openai
import argparse
from flask import Flask, render_template, request, jsonify
import json
import animeCreator
from animeCreator import AnimeBuilder, getFilename
import uuid
from flask_ngrok2 import run_with_ngrok
import dataset

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create_summary', methods=['POST'])
def create_summary():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = animeBuilder.create_novel_summary(story_objects)
    return jsonify(novel_summary)


@app.route('/create_characters', methods=['POST'])
def create_characters():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = animeBuilder.create_characters(story_objects, novel_summary)
    return jsonify(characters)


@app.route('/create_chapters', methods=['POST'])
def create_chapters():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    num_chapters = int(data.get("numChapters", 3))
    chapters = animeBuilder.create_chapters(
        story_objects, novel_summary, characters, num_chapters, nTrials=nTrials)
    return jsonify(chapters)


@app.route('/create_scenes', methods=['POST'])
def create_scenes():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    chapters = data.get("chapters", [])
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))
    all_scenes = animeBuilder.create_scenes(
        story_objects, novel_summary, characters, chapters, num_chapters, num_scenes, nTrials=nTrials)
    return jsonify(all_scenes)


movies = {}


@app.route('/create_movie', methods=['POST'])
def create_movie():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get('novelSummary')
    characters = data.get('characters')
    chapters = data.get('chapters')
    all_scenes = data.get('scenes')
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))

    movie_id = getFilename("", "mov")
    # movie_generator = animeBuilder.generate_movie_data(novel_summary, characters, chapters, scenes)
    movie_generator = animeBuilder.generate_movie_data(
        story_objects, novel_summary, characters, chapters, all_scenes, num_chapters, num_scenes,
        aggressive_merging=aggressive_merging,
        portrait_size=portrait_size)
    movies[movie_id] = MovieGeneratorWrapper(movie_generator,movie_id)

    # return jsonify({"movie_id": movie_id})
    return jsonify(movie_id)

import queue

class MovieGeneratorWrapper:
    def __init__(self, generator, movie_id):
        self.generator = generator
        self.movie_id = movie_id
        self.current_count = 0
        self.available_count = 0
        self.queue_size = 5
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        self._queue = queue.Queue(self.queue_size)
        self._fetch_next_element()

        # Create a table for movie elements
        self.movie_elements_table = db['movie_elements']

    def _get_next_element(self):
        try:
            element = next(self.generator)
            if "image" in element:
                fName = getFilename(savePath, "png")
                element["image"].save(fName)
                element["image"] = fName

            # Add movie_id and count to the element
            element["movie_id"] = self.movie_id
            element["count"] = self.available_count

            # Increment the available count
            self.available_count += 1

            # Insert the element as a new record in the database
            self.movie_elements_table.insert(element)

            return element
        except StopIteration:
            return None
        
    def _fetch_next_element(self):
        self._executor.submit(self._fetch_and_enqueue_next_element)

    def _fetch_and_enqueue_next_element(self):
        while self.available_count - self.current_count < self.queue_size:
            element = self._get_next_element()
            if element is None:
                self._queue.put(None)
                break
            self._queue.put(element)

    def get_next_element(self, count=None):
        if count is not None:
            self.current_count = count

        current_element = self.movie_elements_table.find_one(movie_id=self.movie_id, count=self.current_count)
        if current_element is None:
            current_element = self._queue.get()

        if current_element is not None:
            if self.available_count - self.current_count < self.queue_size:
                self._fetch_next_element()

            current_element = {k: v for k, v in current_element.items() if v is not None}

        self.current_count += 1

        return current_element
    

# DatabaseMovieGenerator class
class DatabaseMovieGenerator:
    def __init__(self, movie_id):
        self.movie_id = movie_id
        self.movie_elements_table = db['movie_elements']
        self.current_count = 0

    def get_next_element(self, count=None):
        if count is not None:
            self.current_count = count

        
        current_element = self.movie_elements_table.find_one(movie_id=self.movie_id, count=self.current_count)
        
        if current_element is None:
            return None

        current_element = {k: v for k, v in current_element.items() if v is not None}

        self.current_count += 1

        return current_element


@app.route('/get_next_element/<string:movie_id>', methods=['GET'])
def get_next_element(movie_id):
    movie_generator = movies.get(movie_id)

    if movie_generator is None:
        # Check if there's at least one element in the movie_elements_table with movie_id
        element_count = db['movie_elements'].count({"movie_id": movie_id})
        
        if element_count == 0:
            return jsonify({"error": "Movie not found"}), 404

        # Create an instance of the DatabaseMovieGenerator class and use it as the movie_generator
        movie_generator = DatabaseMovieGenerator(movie_id, db['movie_elements'])

    count = request.args.get('count', None)
    if count is not None:
        count = int(count)
        print("count",count)

    element = movie_generator.get_next_element(count)
    if element is None:
        return jsonify({"done": "No more elements"}), 200

    return jsonify(element)


@app.route('/get_all_movies', methods=['GET'])
def get_all_movies():
    # Find all movie elements with "debug": "new movie"
    movie_elements = list(db['movie_elements'].find(debug="new movie"))

    # Extract the movie information (title, summary, etc.) from the movie elements
    movies_list = []
    for element in movie_elements:
        movie_info = {
            "movie_id": element["movie_id"],
            "title": element["title"],
            "summary": element["summary"],
        }
        movies_list.append(movie_info)

    return jsonify(movies_list)


@app.route('/movies')
def movie_list():
    return render_template('movie_list.html')

@app.route('/movie/<string:movie_id>', methods=['GET'])
def movie_page(movie_id):
    # Replace 'movie_template.html' with the name of your movie template file
    return render_template('movie_template.html', movie_id=movie_id)

openai.api_key = os.environ['OPENAI_API_KEY']

if __name__ == '__main__':

    savePath = "./static/samples/"

    parser = argparse.ArgumentParser(
        description="Flask App with model name parameter")
    parser.add_argument('--modelName', type=str,
                        default="andite/anything-v4.0", help="Name of the model")
    parser.add_argument('--promptSuffix', type=str,
                        default=", anime drawing", help="add to image prompt")

    parser.add_argument('--negativePrompt', type=str,
                        default="collage, grayscale, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
                        help="negative prompt")

    parser.add_argument('--extraTemplatesFile', type=str,
                        default=None,
                        help="file with template overrides")

    parser.add_argument('--ntrials', type=int, default=5,
                        help='Number of trials (default: 5)')
    
    parser.add_argument('--numInferenceSteps', type=int, default=15,
                        help='Number of inference steps (default: 15)')

    parser.add_argument('--disable-aggressive-merging',
                        action='store_true', help='Disable aggressive merging')

    parser.add_argument('--img2img', action='store_true',
                        help='upscale with img2img')

    parser.add_argument('--ngrok', action='store_true',
                        help='use ngrok tunnel')

    args = parser.parse_args()

    nTrials = args.ntrials

    if args.disable_aggressive_merging:
        aggressive_merging = False
    else:
        aggressive_merging = True

    if args.img2img:
        portrait_size = 256
    else:
        portrait_size = 128


    #database
    db = dataset.connect('sqlite:///movie_elements.db')


    animeBuilder = AnimeBuilder(num_inference_steps=args.numInferenceSteps,
                                textModel="GPT3",
                                diffusionModel=args.modelName,
                                doImg2Img=args.img2img,
                                negativePrompt=args.negativePrompt,
                                suffix=args.promptSuffix,
                                )

    if args.extraTemplatesFile:
        with open(args.extraTemplatesFile, "r") as file:
            code = file.read()
            templateOverrides = eval(code)
            for k, v in templateOverrides.items():
                animeBuilder.templates[k] = v

    if args.ngrok:
        run_with_ngrok(app, auth_token=os.environ["NGROK_TOKEN"])
        app.run()
    else:
        app.run(debug=True, use_reloader=False)

        #app.run()