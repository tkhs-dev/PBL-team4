cd /d %~dp0
pip3 install -r ../requirements.txt
python ../setup.py
start python ./sneak/main.py --auto_close
timeout 1
..\rules\battlesnake.exe play -W 6 -H 6 --name 'Local Solo Test' --url http://localhost:8000 -g solo --browser
