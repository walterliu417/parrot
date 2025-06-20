from parakeet import *

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import os
import pickle
from multiprocessing import Process

# Set up configuration.
helperfuncs.log = False
helperfuncs.datagen = True
helperfuncs.temperature = 30

MOVETIME = 5

def upload_file_to_drive(filename):

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('auth/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': filename, "parents": ["1Jd9V5WT_O5kixA3wqYE_uHb95pYUfnJb"]}
    media = MediaFileUpload(filename, mimetype='application/octet-stream')

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File uploaded. ID: {file.get("id")}')

def datagen(worker):
    print(f"Starting process on worker {worker}")
    engine = Parakeet()
    board = chess.Board()
    boardlist, evallist = [], []
    upload_count = 0
    while upload_count < 1000:
        engine.set_fen(board.fen())
        move, blist, elist = engine.search(MOVETIME, 0, 0)
        boardlist += blist
        evallist += elist
        print(f"{len(evallist)} from worker {worker} currently in storage")
        if len(evallist) > 4096:
            filen = f"selfplay1_{worker}_{upload_count}.chess"
            pickle.dump([boardlist[:4096], evallist[:4096]], open(filen, "wb"))
            boardlist, evallist = [], []
            upload_file_to_drive(filen)
            os.remove(filen)
            print(f"Uploaded file {filen}")
        board.push(move)
        if lt5(board) or board.is_game_over(claim_draw=True):
            board = chess.Board()
            print(f"Starting new game at worker {worker}")



if __name__ == '__main__':
    num_workers = os.cpu_count()

    processes = []
    for i in range(num_workers):
        p = Process(target=datagen, args=(i, ))
        p.start()
        print(f"Started process {i}")
        processes.append(p)

    for p in processes:
        p.join()

    print("All positions generated.")