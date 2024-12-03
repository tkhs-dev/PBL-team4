# 学習管理APIサーバー
## 概要
学習管理APIサーバーは、モデル,学習タスク,学習結果の管理を行う。
Kotlin + Ktorで実装し, PostgreSQLによるデータ永続化を行う。
Dockerコンテナでの実行を想定している。
## キーワード
- REST API
- Kotlin
- Ktor
- PostgreSQL
- Docker

## API仕様
### 認証
すべてのAPIは認証が必要である。
認証は、配布したBearerトークンをヘッダーに設定することで行う。
### エンドポイント
#### 学習
- `GET /train/next` 次の学習タスクを取得
- `POST /train/register` 学習結果を登録
- `POST /train/refresh` 学習の期限を更新
- `GET /train/history` 学習履歴を取得
#### モデル
- `GET /model` モデル一覧を取得
- `GET /model/{id}` モデル詳細を取得

# 実装の詳細