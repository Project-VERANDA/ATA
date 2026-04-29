# Audio-Transcript-Anonymizer

This Pipeline accepts an audio or video file, transcribes the content using WhisperX and applies speaker diarization via Pyannote.
It can be used for interviews, therapy sessions or conversations involving multiple speakers in general.


# Features

Isolation of audio from video files

Automatic transcription via WhisperX

Speaker diarization via Pyannote.audio

Text anonymization with BERT via https://huggingface.co/deryaerman/mmbert_multilingual_pii_ner (Master's thesis project @ the German Research Centre for Artificial Intelligence)

LLM rewriting of text to remove indirect identifiers that BERT will not detect.

Conversation back into synthetic audio,


# Installation

ATA_Setup.sh is a bash script that will install all the relevant files including for the required models. Read the sub-steps below **FIRST**. 

When you have completed these steps, run the setup by typing into your command line while in the ATA folder: ./ATA_Setup.sh
There are a few points where you may be prompted to chose installation features

Upon completion of the script, you will see there is an instruction telling you to update the .env environment file. Only do this if you want to use the LLM API call features of this tool. _I advise ignoring this for now - this feature is not yet fully functional._

## Hardware requirements

This tool allows for flexible model usage. In total with all models, you should ensure you have ~30 GB of storage space available on your system for all features and models. Please note this does _not_ include the use of local LLM models, which this tool does not currently support. This also does not include the storage space required to store all of your audio/video files and the resulting 

WhisperX (for transcription) and pyanote (the speaker-identification model) can be run using GPU (nvidia-only!) or CPU. GPU availability allows for both of these steps to run _considerably_ faster (x13 times faster by one calculation). If there is no GPU available or at least none detected, these processes will fall back to CPU. They will complete faster with more CPU resources.

## Pre-installation
### Model Download / Access
The WhisperX (AKA Faster Whisper) and anonymization model downloads are fully automated. However, the diarization model is gated on Huggingface - You must provide contact information to be able to use or download it. To do this, create a Huggingface account or log in, then navigate to https://huggingface.co/pyannote/speaker-diarization-community-1. Fill in the requested information on the webpage to gain access to the model. 

We highly suggest using the large-v3 model for best accuracy in the transcription. If your system does not have a GPU available and limited CPU resources, you may wish to consider a smaller model. Unfortunately, I cannot give you a guide about which model will work best for your system.


### Huggingface Tokens
While logged in, navigate to https://huggingface.co/settings/tokens and create a read-only token. Save the token code somewhere (for good data security practises, don't save it on the cloud unless your cloud is encrypted! For example, with Proton (link)).

When you run the self-install script, you will be prompted for the huggingface token. Paste the token when prompted.

## Flask Installation

You will be prompted with a question asking if you want to install the Flask requirements. This is only necessary if you also want to enable the web interface. If you do not need the web interface, input `n` and press enter. Otherwise, input `y` and press enter. The required packages will install.


# Running the script

After the ATA_SelfInstall script has completed with your desired options, you can use ATA either via command-line arguments or via the web interface.
I strongly suggest regularly pulling the latest version of this software with the following command in the ATA directory.
```git pull```

## Commandline Processing of data
Run the script. From the ATA folder, in your command-line, type: python pipeline/process.py 

### LLM rewriting of files.
This version of the software currently attempts to call a Charité LLM server. This will not work for anyone else - it only works on ATA's internal Charité server. When running this on your own instance, you will have to update the API endpoint and API token for your chosen service provider. Edit the hidden environment file (.env) file with your API key, and update the API endpoint in pipeline/process.py and web_interface/app.py for your chosen LLM API provider.
**I suggest for now ignoring this feature.**

If you do have an LLM provider available, then you can use the LLM-rewrite feature in the command line with
```python pipeline/process.py --enable-llm```
This will output both the BERT-anonymized texts in the annoym folder in /pipeline/, and the LLM-rewritten texts in the LLM-Anon folder in /pipeline/. The LLM rewrite rewrites the BERT-anonymized texts, not the non-anonymized transcripts.

## Web interface set-up
To use the web interface, you need to install the flask interface prompted during the ATA_SelfInstall.sh set-up outlined above. If you did not install it originally, running the script again and selecting the flask install on the new run should work fine. 
**Back up your files before re-running the 

To start the web interface, type instead: python interactive_app/app.py
You will hopefully be able to access the web interface by opening https://10.0.1.159/ in your browser. HTTPS is required to enable the microphone recording feature.
If you are able to access the web interface, it should hopefully be fairly self-explanatory. Record a sample audio or upload your own. Click the buttons.
If the web interface is not accessible, you will need to do some troubleshooting. This repo is not advanced enough to handle that entirely internally. Sorry.

### LLM Models
This version of the software currently attempts to call a Charité LLM server. This will not work for anyone else - it only works on ATA's internal Charité server. When running this on your own instance, you will have to update the API endpoint and API token for your chosen service provider. Edit the hidden environment file (.env) file with your API key, and update the API endpoint in pipeline/process.py and web_interface/app.py for your chosen LLM API provider.
**I suggest for now ignoring this feature.**

# Post-installation model downloading
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

# Pending updates

* [x] Re-adding the anonymization process.
* [ ] Adding the models to the Github Repo.
* [x] Add a web interface.
  *   [ ] Update the existing web interface to be a "demo" interface.
  *   [ ] Create a new production interface.
      * [ ] Add buttons for the interface to enable/disable each feature in the pipeline.
* [x] Add a recording button to the web interface for demos.
* [ ] Add batch transcripting via selected folder instead of only with selected files.
* [ ] Add automatic cleanup of the input video/audio folders. 
* [ ] Add arguments for a debug running of the pipeline.
* [x] Add text to speech for the output transcripts.
* [x] Switch the diarization to speaker-diarization-community-1 from the older diarization model.
* [x] Automate the downloading of WhisperX models.
* [ ] Switch from API calls for the AI models to local model processing.
  *  [ ] Add a check/installation for local models, and if not, fall back to API calls.
  *  [ ] Disable the LLM rewrite function when no LLM model is available.
  *  [ ] Switch the .env file to also host the LLM API endpoint variable, instead of hardcoding it into the code.
* [ ] Do some optimizations for the LLM system prompt.
* [ ] Try getting all the local LLM models working.
* [x] Update the Running the Script section of this ReadMe.
* [ ] Add arguments for process.py to enable/disable each feature in the pipeline.
* [ ] Add code enabling other non-nvidia GPUs.
* [ ] Switch from Google TTS to a new, locally-running text-to-speech tool.
* [ ] Disable the translation feature of WhisperX.
* [ ] Look into whether there can be chunk-processing of live-recorded data for live transcription and anonymization.



* [ ]

# Future additions

* Improved dialogue and accent handling.
* K-Anonymity for databases or folders.
