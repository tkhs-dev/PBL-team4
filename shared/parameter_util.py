#　必要なパラメーターを取得する関数を定義する
# example:
#  def get_snake_length(game_state: dict) -> int:
#      return game_state["you"]["length"]



#蛇の長さ
def get_snake_length(game_state: dict) -> int:
    length = game_state["you"]["length"]

    return length

#ヘルス
def get_snake_health(game_state: dict) -> int:
    health = game_state["you"]["health"]

    return health