from flask import Flask, render_template, request, send_file
import os
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel
import json
import cv2
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import *
app = Flask(__name__)
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["filedata"]
mycol = db["file_data"]
now = datetime.now()
# Define a directory to store uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'downloads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), "output")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# Ensure the upload and download directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
def add_audio_to_video(video_path, audio_path, output_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set the video clip's audio to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video with added audio
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile="temp_audio.m4a", remove_temp=True)

def add_captions(video_path, json_path, output_path):
    # Read the video
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read subtitles from the JSON file
    with open(json_path, 'r') as file:
        subtitles_data = json.load(file)

    # Process each frame and add subtitles
    current_subtitle_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_subtitle_index < len(subtitles_data):
            current_subtitle = subtitles_data[current_subtitle_index]

            if current_time >= current_subtitle["start"] and current_time <= current_subtitle["end"]:
                # Display the whole line with white text and red border, and change text color for the word being spoken
                whole_line = current_subtitle["word"]
                for content in current_subtitle["textcontents"]:
                    if current_time >= content["start"] and current_time <= content["end"]:
                        word_to_highlight = content["word"]
                        word_start = whole_line.find(word_to_highlight)
                        word_end = word_start + len(word_to_highlight)

                        font_scale = width / 800.0  # Adjust the constant as needed
                        font_thickness = int(2 * font_scale)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(whole_line, font, font_scale, font_thickness)[0]
                        text_x = (width - text_size[0]) // 2
                        text_y = height - int(30 * font_scale)

                        # Display the whole line with white text and red border
                        cv2.putText(frame, whole_line, (text_x, text_y), font, font_scale, (0, 0, 255), int(font_thickness *5), cv2.LINE_AA)
                        cv2.putText(frame, whole_line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                        # Change the text color of the word being spoken
                        cv2.putText(frame, whole_line[:word_start], (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        cv2.putText(frame, whole_line[word_start:word_end], (text_x + cv2.getTextSize(whole_line[:word_start], font, font_scale, font_thickness)[0][0], text_y),
                                    font, font_scale, (255, 0, 0), int(font_thickness *6), cv2.LINE_AA)
                        cv2.putText(frame, whole_line[word_start:word_end], (text_x + cv2.getTextSize(whole_line[:word_start], font, font_scale, font_thickness)[0][0], text_y),
                                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        cv2.putText(frame, whole_line[word_end:], (text_x + cv2.getTextSize(whole_line[:word_end], font, font_scale, font_thickness)[0][0], text_y),
                                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Move to the next subtitle if the current time has passed the end time
            if current_time > current_subtitle["end"]:
                current_subtitle_index += 1

        # Write the frame to the output video
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()




def split_text_into_lines(data):

    MaxChars = 30
    #maxduration in seconds
    MaxDuration = 2.5
    #Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0


    for idx,word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)


        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx>0:
          gap = word_data['start'] - data[idx-1]['end']
          # print (word,start,end,gap)
          maxgap_exceeded = gap > MaxGap
        else:
          maxgap_exceeded = False


        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0


    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)

    return subtitles
@app.route('/')
def index():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return render_template('home.html')
@app.route('/video_to_video')
def vtv():
    return render_template("vtv.html")
@app.route('/upload_vtv', methods=['POST'])
def upload_file_vtv():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output = os.path.join(app.config['OUTPUT_FOLDER'],file.filename)
        video_to_video(filepath,output)
        
        mydict = {"name":file.filename,"operation" : "add captions to video","time": now.strftime("%H:%M:%S") }
        x = mycol.insert_one(mydict) 
        # Check if the file was saved successfully
        if os.path.exists(filepath):
            return render_template('upload_success_vtv.html', filename=file.filename)
        else:
            return "Failed to upload " + file.filename
@app.route('/video_to_audio')
def vta():
    return render_template("vta.html")
@app.route('/upload_vta', methods=['POST'])
def upload_file_vta():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        clip = mp.VideoFileClip(filepath)
        file_name = file.filename.split(".")[0]
        output = os.path.join(app.config['OUTPUT_FOLDER'],file_name+".wav")
        clip.audio.write_audiofile(output)
        mydict = {"name":file.filename,"operation" : "extract audio from video","time": now.strftime("%H:%M:%S") }
        x = mycol.insert_one(mydict) 

        # Check if the file was saved successfully
        if os.path.exists(filepath):
            return render_template('upload_success_vta.html', filename=file.filename)
        else:
            return "Failed to upload " + file.filename
@app.route('/download/<filename>')
def download_file(filename):
    # Generate the path to the file
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    # Check if the file exists before trying to send it
    if os.path.exists(filepath):
        if(os.path.exists(os.path.join(app.config['DOWNLOAD_FOLDER'],filename))):
            os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'],filename))
        os.rename(filepath, os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
        # Send the file as an attachment
        return send_file(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), as_attachment=True)
    else:
        return "File not found"
@app.route("/download/<filename>/audio")
def download_audio(filename):
     # Generate the path to the file
    file_name = filename.split(sep=".")[0]+'.wav'
    print(file_name)
    filepath = os.path.join(app.config['OUTPUT_FOLDER'],file_name)
    print(filepath)
    # os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'],"output_audio.wav"))
    # Check if the file exists before trying to send it
    if os.path.exists(filepath):
        # Move the file to the download folder
        if(os.path.exists(os.path.join(app.config['DOWNLOAD_FOLDER'],file_name))):
            os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'],file_name))
        os.rename(filepath, os.path.join(app.config['DOWNLOAD_FOLDER'],file_name))
        # Send the file as an attachment
        return send_file(os.path.join(app.config['DOWNLOAD_FOLDER'],file_name), as_attachment=True)
    else:
        return "File not found"
@app.route("/upload_vts", methods=["POST"])
def upload_file_vts():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        
        file_name = file.filename.split(".")[0]
        output = os.path.join(app.config['OUTPUT_FOLDER'],file_name+".srt")
        video_to_srt(filepath,output)
        mydict = {"name":file.filename,"operation" : "get srt from video","time": now.strftime("%H:%M:%S") }
        x = mycol.insert_one(mydict) 
        # Check if the file was saved successfully
        if os.path.exists(filepath):
            return render_template('upload_success_vts.html', filename=file.filename)
        else:
            return "Failed to upload " + file.filename
@app.route("/download/<filename>/srt")
def download_srt(filename):
     # Generate the path to the file
    file_name = filename.split(sep=".")[0]+'.srt'
    print(file_name)
    filepath = os.path.join(app.config['OUTPUT_FOLDER'],file_name)
    print(filepath)
    # os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'],"output_audio.wav"))
    # Check if the file exists before trying to send it
    if os.path.exists(filepath):
        
        # Send the file as an attachment
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'],file_name), as_attachment=True)
    else:
        return "File not found"
@app.route("/download/<filename>/vtvt")
def download_vtvt(filename):
    file_name = filename.split(sep=".")[0]+".mp4"
    filepath = os.path.join(app.config['OUTPUT_FOLDER'],file_name)
    print(filepath)
    # os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'],"output_audio.wav"))
    # Check if the file exists before trying to send it
    if os.path.exists(filepath):
        
        # Send the file as an attachment
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'],file_name), as_attachment=True)
    else:
        return "File not found"           
def video_to_video(input_path,output_path):
    clip = mp.VideoFileClip(input_path)
    clip.audio.write_audiofile(r"converted_mp3.wav")
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(r"converted_mp3.wav", word_timestamps=True)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))   
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': str(word.word), 'start': word.start, 'end': word.end}) 
    with open('data.json', 'w') as f:
        json.dump(wordlevel_info, f,indent=4) 
    with open('data.json', 'r') as f:
        wordlevel_info_modified = json.load(f)  
    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)
    print (linelevel_subtitles) 
    for line in linelevel_subtitles:
        json_str = json.dumps(line, indent=4)
        print(json_str) 
    with open('data.json', 'w') as f:
        json.dump(linelevel_subtitles, f,indent=4)
        
    add_captions(input_path,"data.json","temp.mp4")  
    add_audio_to_video("temp.mp4","converted_mp3.wav",output_path)    
@app.route("/video_to_srt") 
def vts():
    return render_template("vts.html")
@app.route("/audio_to_srt") 
def ats():
    return render_template("ats.html")
@app.route("/upload_ats", methods=["POST"])
def upload_file_ats():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        
        file_name = file.filename.split(".")[0]
        output = os.path.join(app.config['OUTPUT_FOLDER'],file_name+".srt")
        audio_to_srt(filepath,output)
        mydict = {"name":file.filename,"operation" : "get srt from audio","time": now.strftime("%H:%M:%S") }
        x = mycol.insert_one(mydict) 
        # Check if the file was saved successfully
        if os.path.exists(filepath):
            return render_template('upload_success_vts.html', filename=file.filename)
        else:
            return "Failed to upload " + file.filename
@app.route("/video_to_video_translated")
def vtvt_Load():
    return render_template("vtvt.html")  
@app.route("/team")
def team():
    return render_template("Team.html")    
@app.route("/about")
def about():
    return render_template("about.html")  
@app.route("/upload_vtvt",methods=["POST"])
def upload_file_vtvt():
        if request.method == 'POST':
        # Check if the post request has the file part
            if 'file' not in request.files:
                return "No file part"
            
            file = request.files['file']
            
            # If the user does not select a file, the browser submits an empty file without a filename
            if file.filename == '':
                return "No selected file"
            
            # Save the uploaded file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            
            file_name = file.filename.split(".")[0]
            output = os.path.join(app.config['OUTPUT_FOLDER'],file_name+"_translated.mp4")
            add_translated_captions(filepath,output)
            # Check if the file was saved successfully
            if os.path.exists(filepath):
                return render_template('upload_success_vtvt.html', filename=file.filename+"_translated")
            else:
                return "Failed to upload " + file.filename 
def audio_to_srt(input_path,output_path): 
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(input_path, word_timestamps=True)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))   
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': str(word.word), 'start': word.start, 'end': word.end}) 
    with open('data.json', 'w') as f:
        json.dump(wordlevel_info, f,indent=4) 
    with open('data.json', 'r') as f:
        wordlevel_info_modified = json.load(f)  
    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)
    print (linelevel_subtitles) 
    for line in linelevel_subtitles:
        json_str = json.dumps(line, indent=4)
        print(json_str) 
    with open('data.json', 'w') as f:
        json.dump(linelevel_subtitles, f,indent=4)
    json_to_srt("data.json",output_path)       
def video_to_srt(input_path,output_path):    
    clip = mp.VideoFileClip(input_path)
    clip.audio.write_audiofile(r"converted_mp3.wav")
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(r"converted_mp3.wav", word_timestamps=True)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))   
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': str(word.word), 'start': word.start, 'end': word.end}) 
    with open('data.json', 'w') as f:
        json.dump(wordlevel_info, f,indent=4) 
    with open('data.json', 'r') as f:
        wordlevel_info_modified = json.load(f)  
    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)
    print (linelevel_subtitles) 
    for line in linelevel_subtitles:
        json_str = json.dumps(line, indent=4)
        print(json_str) 
    with open('data.json', 'w') as f:
        json.dump(linelevel_subtitles, f,indent=4)
    json_to_srt("data.json",output_path)
def json_to_srt(json_filepath, output_filepath):
    import json
    
    # Load JSON data
    with open(json_filepath, 'r') as json_file:
        data = json.load(json_file)
    
    # Open output file for writing SRT
    with open(output_filepath, 'w') as srt_file:
        index = 1  # SRT index counter
        for entry in data:
            start_time = int(entry['start'] * 1000)  # Convert to milliseconds
            end_time = int(entry['end'] * 1000)      # Convert to milliseconds

            # Write index
            srt_file.write(str(index) + '\n')

            # Write time frame
            srt_file.write(f"{milliseconds_to_srt_time(start_time)} --> {milliseconds_to_srt_time(end_time)}\n")

            # Write text contents
            for item in entry['textcontents']:
                srt_file.write(item['word'] + ' ')
            srt_file.write('\n\n')

            index += 1
def add_translated_captions(input_file,output_file):
    clip = mp.VideoFileClip(input_file)
    clip.audio.write_audiofile(r"converted_mp3.wav")
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(r"converted_mp3.wav", word_timestamps=True)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': str(word.word), 'start': word.start, 'end': word.end})    
    with open('data.json', 'w') as f:
        json.dump(wordlevel_info, f,indent=4)      
    with open('data.json', 'r') as f:
        wordlevel_info_modified = json.load(f)      
    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)      
    for line in linelevel_subtitles:
     json_str = json.dumps(line, indent=4)
     print(json_str)
    with open('data.json', 'w') as f:
        json.dump(linelevel_subtitles, f,indent=4) 
    hindi_text=""    
    for data in linelevel_subtitles:
        hindi_text = hindi_text + data["word"]
    # def to_markdown(text):
    #     text = text.replace('•', '  *')
    #     return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    # GOOGLE_API_KEY="AIzaSyB1nZldtO37q43RqcnwerYN6xK0aNEMbuY"

    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content("translate this to hindi"+hindi_text)
    # # to_markdown(response.parts)    
    # translator = Translator(to_lang="hi")
    
    
    
    # # This module is imported so that we can  
    # # play the converted audio 
    # # The text that you want to convert to audio 
    # mytext = translator.translate(hindi_text)
    def translate_to_hindi(text):
        try:
            translation = GoogleTranslator(source='en', target='hi').translate(text)
            return translation
        except Exception as e:
            return str(e)  
    mytext = translate_to_hindi(hindi_text)
    # Language in which you want to convert 
    language = 'hi'
    
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    
    # Saving the converted audio in a mp3 file named 
    # welcome  
    myobj.save("output2.mp3")    
    
    video_path = input_file
    audio_path = "output2.mp3"
    output_path = "Intro3.mp4"

    add_audio_to_video(video_path, audio_path, output_path)
    # loading video dsa gfg intro video 
    clip1 = VideoFileClip("Intro3.mp4") 
    
    # getting only first 5 seconds 
    
    # applying speed effect 
    final = clip1.fx(vfx.speedx, clip1.duration/clip.duration) 
    
    # showing final clip 
    final.ipython_display() 
    final.audio.write_audiofile("output2.mp3")
    segments, info = model.transcribe(r"output2.mp3", word_timestamps=True)
    segments = list(segments)  # The transcription will actually run here.
    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': str(word.word), 'start': word.start, 'end': word.end}) 
    with open('translated.json', 'w') as f:
        json.dump(wordlevel_info, f,indent=4)   
    with open('translated.json', 'r') as f:
        wordlevel_info_modified = json.load(f)  
    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)
    for line in linelevel_subtitles:
        json_str = json.dumps(line, indent=4)
        print(json_str)
    with open('translated.json', 'w') as f:
        json.dump(linelevel_subtitles, f,indent=4)
        def add_captions_hindi(video_path, json_path, output_path):
        # Read the video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Read subtitles from the JSON file
            with open(json_path, 'r', encoding='utf-8') as file:
                subtitles_data = json.load(file)

            # Process each frame and add subtitles
            current_subtitle_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if current_subtitle_index < len(subtitles_data):
                    current_subtitle = subtitles_data[current_subtitle_index]

                    if current_time >= current_subtitle["start"] and current_time <= current_subtitle["end"]:
                        # Display the whole line with white text and red border, and change text color for the word being spoken
                        whole_line = current_subtitle["word"]
                        for content in current_subtitle["textcontents"]:
                            if current_time >= content["start"] and current_time <= content["end"]:
                                word_to_highlight = content["word"]
                                word_start = whole_line.find(word_to_highlight)
                                word_end = word_start + len(word_to_highlight)

                                font_scale = width / 800.0  # Adjust the constant as needed
                                font_thickness = int(2 * font_scale)
                                
                                # Change the font to support Hindi characters
                                hindi_font_path = "Arya-Regular.ttf"
                                hindi_font_size = int(30 * font_scale)
                                hindi_font = ImageFont.truetype(hindi_font_path, hindi_font_size)

                                # Create an image with the Hindi text
                                img = Image.fromarray(frame)
                                draw = ImageDraw.Draw(img)
                                draw.text((10, height - 40), whole_line, font=hindi_font, fill=(255, 255, 255))

                                # Convert the image back to a NumPy array
                                frame = np.array(img)

                    # Move to the next subtitle if the current time has passed the end time
                    if current_time > current_subtitle["end"]:
                        current_subtitle_index += 1

                # Write the frame to the output video
                out.write(frame)

            # Release video capture and writer
            cap.release()
            out.release()

        f = input_file.split(".")[0]

        video_path = input_file
        json_path = "translated.json"
        output_path = output_file

        add_captions_hindi(video_path, json_path, output_path)                      
def milliseconds_to_srt_time(milliseconds):
    seconds = milliseconds // 1000
    milliseconds %= 1000
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if __name__ == '__main__':
    app.run()
