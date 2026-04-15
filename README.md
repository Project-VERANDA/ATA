# Audio-Transcript-Anonymizer

This Pipeline accepts an audio or video file, transcribes the content using WhisperX and applies speaker diarization via Pyannote.
It can be used for interviews, therapy sessions or conversations involving multiple speakers in general.


# Features

Audio/video (mp3/mp4) input
Automatic transcription via WhisperX
Speaker diarization via Pyannote.audio


# Installation

To run the transcription pipeline you'll need Python 3.10. and Anaconda.

ATA_Setup.sh is a bash script that will install all the relevant files EXCEPT for the required models. Run the setup by typing into your command line: ./ATA_Setup.sh

This script requires models--Systran--faster-whisper-large-v3 and pyannote-SpeakerDiarization (https://huggingface.co/pyannote/speaker-diarization) to be downloaded to the pipeline/model/ folder. These must be downloaded separately from Huggingface after accepting the pre-requisite agreements and providing the requested information.

It may not be necessary to install the whisper-large-v3 file to the models folder, as the bash install script. This should be tested.


# Running the script

Open the pipeline script and update all fields marked with *** and save your changes. 
Run the script. In your command-line, type: python process.py 


# Pending updates

* Re-adding the anonymization process.
** This is pending the addition of anonymization models.

* Adding the models to the Github Repo.
* Add a web interface.