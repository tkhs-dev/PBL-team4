# PBL-Team4
## 主なディレクトリ構成
```
PBL-Team4
│  README.md #このファイル
│  .gitignore #gitの無視リスト,必要に応じて追記
│  setup.py #battlesnakeの環境を自動で構築するスクリプト
│  requirements.txt #プログラムで必要なライブラリを記述するファイル
├─solo
│  │  run_solo.bat　#soloをローカルで実行するバッチファイル
│  └─sneak
│      │  main.py #soloのメインプログラム
│      └─ server.py #soloのサーバープログラム,基本触らない
├─duel #今後追加
└─rules #setup.pyを実行すると生成される.battlesnakeの実行ファイルがある

```

## Run Locally
Soloのプログラムをローカルで簡単に実行できます。
```bash
.\solo\run_solo.bat
```
> [!NOTE]
> 終了後にpythonのウィンドウが残ります。もう一度実行する際は、前のウィンドウを閉じてから実行してください。