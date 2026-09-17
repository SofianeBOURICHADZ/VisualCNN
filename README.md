# VisualCNN!

Hello everyone, **VisualCNN** is an AI offline, open-source application that relies on python on PyTorch to serve as a tool for learning, experimenting and automation. It allows the usage of a Graphical User Interface (GUI) in order to design, tune, train and help deploy AI Image Classifiers (Other versions for other tasks available soon) . This allows to teach fundamentals of CNN without writing code.


# Why did I make this

I made this application to make CNN designing quicker, faster, and allows automation without touching code and having to make commits or backups for files. And my solution for this was VisualCNN.

## Features

- Design a CNN from scratch, including both the convolutional layers and the classification head.
- Automatically load a dataset from a folder and prepare it for training and validation.
- Setting up training, validation, LR scheduling and model saving without touching code and quickly.
- Automatically see metrics pop-up on the table and have graphs of changes of loss and accuracy metrics over epochs.
- The ability to use GPU for CUDA supported graphics cards.
- Deploy model and have it ready for usage in next projects. It even includes the code that you would write if you did the model design manually!

## How to run
**Case 1: Python isn't installed on your host machine / No Command Line (Recommended):**
-Go to the Releases page on the left of this GitHub repository and download the prebuilt portable app for your specific OS.
-Open the executable file you just downloaded.
**Case 2: Python is installed on your host machine**:
-Clone this repository with `git clone https://github.com/SofianeBOURICHADZ/VisualCNN`.
-Access the folder via `cd VisualCNN`.
-Install the required packages to run this app via `pip install -r requirements.txt` (You may need to use another command for installation, depending on your setup, but this is the default one).
-Run with `python3 VisualCNN.py`



## Building portable app to a target OS:
To build the portable app for your target OS. Follow these steps:
- Use a computer that is running your target OS.
- Install the requirements via `pip install -r requirements.txt` (You may need to use another command for installation, depending on your setup, but this is the default one).
- Install PyInstaller via this command `pip install pyinstaller`. (You may need to use another command for installation, depending on your setup, but this is the default one).
- Run this command `pyinstaller --clean --onefile --copy-metadata torch --copy-metadata torchvision --collect-all torchvision VisualCNN.py` to start the building process.
- Run `cd build` then if you are on Linux run `chmod +x VisualCNN` to make the file executable, if you are on Mac then run `xattr -cr VisualCNN.app`
- Run the app via `./VisualCNN`

## AI Agents and LLM Assistance Disclosure - IMPORTANT- TRANSPARENCY STATEMENT:
I used Gemini 3.6 Flash for the GUI code and also part of the deployment . I independently designed and implemented the data pipeline, augmentation, CNN architecture, training and evaluation loops, checkpointing, inference, input validation, and model-related logic. I reviewed and slightly modified the generated integration code to connect it to my implementation.
