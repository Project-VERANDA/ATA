# Audio-Transcript-Anonymizer

This Pipeline accepts an audio or video file, transcribes the content using WhisperX and applies speaker diarization via Pyannote.
It can be used for interviews, therapy sessions or conversations involving multiple speakers in general.


# Features

Audio/video (mp3/mp4) input
Automatic transcription via WhisperX
Speaker diarization via Pyannote.audio


# Installation

To run the transcription pipeline you'll need Python 3.10. and Anaconda.

ATA_Setup.sh is a bash script that will install all the relevant files EXCEPT for the required models.

This script requires models--Systran--faster-whisper-large-v3 and pyannote-SpeakerDiarization (https://huggingface.co/pyannote/speaker-diarization) to be downloaded to the pipeline/model/ folder. These must be downloaded separately from Huggingface after accepting the pre-requisite agreements and providing the requested information.

It may not be necessary to install the whisper-large-v3 file to the models folder, as the bash install script. This should be tested.


# Running the script
Place the script into a folder along with the subfolders 'audios' (for mp3) and/or 'videos' (for mp4) and add your media to the respective folder.

Open the script and update all fields marked with *** and save your changes. 
Run the script. 
