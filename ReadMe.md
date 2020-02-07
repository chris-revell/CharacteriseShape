# How To Run `CharacteriseShape.py`

These instructions assume running on macOS.

## Install Packages

1. If you already have python3 and numpy, scipy, matplotlib, astropy, openCV, and scikit-image, skip this section.
2. Open the macOS Terminal app.
3. Install Homebrew package manager https://brew.sh - enter the following in the command line `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
3. Install python3 by running the following in the terminal `brew install python3`
4. Install required packages by entering the following: `pip3 install numpy; pip3 install matplotlib; pip3 install astropy; pip3 install scipy; pip3 scikit-image; pip3 install opencv-python`.

## Run script

1. Open the Terminal application.
2. Change working directory to the location of the CharacteriseShape.py script. Eg. if the script is located at `/Users/christopher/Desktop/CharacteriseShape`, enter `cd /Users/christopher/Desktop/CharacteriseShape`.
3. Identify the location of the image file to be analysed. Eg. `/Users/christopher/Desktop/image.tif`.
4. Run the script by entering the following: `python3 CharacteriseShape.py <filename>`. For example, `python3 CharacteriseShape.py /Users/christopher/Desktop/image.tif`
