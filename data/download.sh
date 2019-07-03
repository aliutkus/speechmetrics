# download evaluation restuls
wget -O vcc2018_listening_test_scores.zip "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3257/vcc2018_listening_test_scores.zip?sequence=1&isAllowed=y" 
unzip vcc2018_listening_test_scores.zip
rm vcc2018_listening_test_scores.zip

# download submitted_systems_converted_speech
wget -O vcc2018_submitted_systems_converted_speech.tar.gz "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz?sequence=10&isAllowed=y" 
tar zxvf vcc2018_submitted_systems_converted_speech.tar.gz
rm vcc2018_submitted_systems_converted_speech.tar.gz

# rename dir
mkdir submit
mv ./mnt/sysope/test_files/testVCC2/*.wav ./submit
rm -r ./mnt

# downsample
python downsample.py
