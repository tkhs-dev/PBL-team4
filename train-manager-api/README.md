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
エンドポイントの詳細はOpenAPIでopenapi.yamlに記述している.
詳細は[こちら](./openapi.yaml)を参照,もしくは,./apidoc.htmlをブラウザで開くことで確認できる。

# 実装の詳細