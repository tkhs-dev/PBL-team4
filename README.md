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

# 学習について
## いじるべき項目
- 個体のスコアの算出方法:trainer/main.pyのevaluate_single関数やmove_callbackで設定.ターン毎と最終結果をもとにスコアを計算している.
- 個体がプレイするゲーム数:trainer/main.pyのevaluate関数で設定している.多いほど学習に時間がかかるが,精度が上がる可能性がある
- 遺伝的学習の各種メソッド:trainer/main.pyのtrain関数で設定,deapの公式ドキュメント参照して,選択、交叉、突然変異のメソッドを変更できる.学習速度に影響を与えるかも
- 遺伝的学習の個体数や世代数:学習速度に大きな影響を与える.小さすぎると学習が偏ることがあるので注意
##  学習方法
solo/tainer内で`python main.py`を実行.
学習結果は,世代の一部が./pth/#######/evaluator-turn_score.pthに保存される.また,各学習の最終結果がevaluator.pthに保存される.基本的には../pth内の一番スコアが大きいものを使えばよい？
以前の学習を引き継ぐ場合は`python main.py pthのパス`を実行.
例`python main.py ./pth/1731046890/evaluator-7_807.pth`

## 学習結果をもとにヘビを動かす
soloディレクトリ直下に学習ファイルをevaluator.pthとして保存する.その後run_solo.batを実行すると,学習結果をもとにヘビが動く(はず).