# Audio-Transcript-Anonymizer

This Pipeline accepts an audio or video file, transcribes the content using WhisperX and applies speaker diarization via Pyannote.
It can be used for interviews, therapy sessions or conversations involving multiple speakers in general.


# Features

Audio/video (mp3/mp4) input
Automatic transcription via WhisperX
Speaker diarization via Pyannote.audio


# Installation

To run the transcription pipeline you'll need Python 3.10. and Anaconda.

ATA_Setup.sh is a bash script that will install all the relevant files including for the required models. Read the sub-steps below FIRST. When you have completed these steps, run the setup by typing into your command line while in the ATA folder: ./ATA_Setup.sh

## Model Download / Access
The WhisperX (AKA Faster Whisper) and anonymization model downloads are fully automated. However, the diarization model is gated on Huggingface - You must provide contact information to be able to use or download it. To do this, create a Huggingface account or log in, then navigate to https://huggingface.co/pyannote/speaker-diarization-community-1. Fill in the requested information on the webpage to gain access to the model. 

We highly suggest using the large-v3 model for best accuracy in the transcription.


## Huggingface Tokens
While logged in, navigate to https://huggingface.co/settings/tokens and create a read-only token. Save the token code somewhere (for good data security practises, don't save it on the cloud unless your cloud is encrypted! For example, with Proton (link)).

When you run the self-install script, you will be prompted for the huggingface token. Paste the token when prompted.



If the model download fails due to a missing token, or you want to install the models at a later time, you can download them with following commands:

Download the community-1 model with the following guide:

First, authenticate to Huggingface. You will need a Huggingface acccount first, and then create an access token (which you can get from https://huggingface.co/settings/tokens after logging in).

```huggingface-cli login```
Then paste your access token.

If your authentication functions correctly, navigate to the ATA/pipeline/model/ folder and download the diarization model with the following command:
```
huggingface-cli download pyannote/speaker-diarization-community-1 \
>   --local-dir models--pyannote--speaker-diarization-community-1 \
>   --local-dir-use-symlinks False
```

Download all the WhisperX models with the following command, which can be copy/pasted into the command line when in the pipeline/model/ folder.
```for size in tiny base small medium large-v2 large-v3; do
  echo "Downloading faster-whisper-$size..."
  huggingface-cli download Systran/faster-whisper-$size \
    --local-dir "models--Systran--faster-whisper-$size" \
    --local-dir-use-symlinks False
done
```

Alternatively, use only the large-v3 model.

```huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir models--Systran--faster-whisper-large-v3 \
  --local-dir-use-symlinks False
  ```

_It may not be necessary to install the whisper-large-v3 file to the models folder. This should be tested._


# Running the script

Open the pipeline script and update all fields marked with *** and save your changes. 
Run the script. From the MAIN folder, in your command-line, type: python pipeline/process.py 
To start the web interface, type instead: python interactive_app/app.py


# Pending updates

* [x] Re-adding the anonymization process.
* [ ] Adding the models to the Github Repo.
* [x] Add a web interface.
* [x] Add a recording button to the web interface for demos.
* [ ] Add batch transcripting via selected folder.
* [ ] Add automatic cleanup of the input video/audio folders. 
* [ ] Add arguments for a debug running of the pipeline.
* [x] Add text to speech for the output transcripts.
* [x] Switch the diarization to speaker-diarization-community-1 from the older diarization model.
* [x] Automate the downloading of WhisperX models.
* [ ] Switch from API calls for the AI models to local model processing.
* [ ] Update the Running the Script section of this ReadMe.

# Future additions

* Improved dialogue and accent handling.
* K-Anonymity for databases or folders.